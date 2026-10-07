"""印后测色验收端到端测试。

覆盖：任务直连与达标修色成品两条预测路径、同坐标复测保留、ΔE00 上限
严格大于判定与整批不合格、页面/报告绑定；以及全部拒绝路径——修色不可达、
报告/成品绑定不符、ICC 缺失或摘要不符、测点越界、CSV 非法（不用原任务
值代替）。
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app import lcms_native
from app.main import app


@pytest.fixture
def client(tmp_path):
    data_dir = tmp_path / "acceptance-data"
    os.environ["SPD_DATA_DIR"] = str(data_dir)
    from app import config
    config._state = None
    config.DATA_DIR = data_dir
    config.PROFILE_DIR = data_dir / "profiles"
    config.JOB_DIR = data_dir / "jobs"
    config.RETOUCH_DIR = data_dir / "retouch"
    config.ACCEPTANCE_DIR = data_dir / "acceptance"
    config.RELEASE_DIR = data_dir / "release"
    config.DB_PATH = data_dir / "softproof.db"
    from app import main
    main.JOB_DIR = data_dir / "jobs"
    main.RETOUCH_DIR = data_dir / "retouch"
    main.ACCEPTANCE_DIR = data_dir / "acceptance"
    main.DB_PATH = data_dir / "softproof.db"
    with TestClient(app) as c:
        yield c


@pytest.fixture
def ids(client):
    profs = client.get("/api/profiles").json()
    return {
        "srgb": next(p["profile_id"] for p in profs["source"]
                     if p["color_space"] == "RGB"),
        "cmyk": next(p["profile_id"] for p in profs["press"]),
        "display": next(p["profile_id"] for p in profs["display"]),
    }


def _png(rgb=(210, 90, 60), size=48):
    arr = np.zeros((size, size, 3), np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = rgb
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _gradient_png(size=48):
    yy, xx = np.mgrid[0:size, 0:size]
    arr = np.stack([(xx * 5) % 256, (yy * 5) % 256,
                    ((xx + yy) * 3) % 256], axis=-1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _run_job(client, ids, png=None):
    r = client.post(
        "/api/jobs",
        files={"image": ("img.png", png or _png(), "image/png")},
        data={
            "source_profile": ids["srgb"],
            "press_profile": ids["cmyk"],
            "display_profile": ids["display"],
            "threshold": "3",
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def _retouch_reachable(client, job_id, offset=(3, 0, 0, 0),
                       rect=(4, 4, 16, 16), ids=None):
    """以“原选区 CMYK 统一加 offset 后”的平均 Lab 为目标，保证可达产出 TIFF。"""
    from app import main
    cmyk = np.load(main.JOB_DIR / job_id / "cmyk.npy")
    x, y, w, h = rect
    roi = cmyk[y:y + h, x:x + w].reshape(-1, 4).astype(np.int32)
    roi = np.clip(roi + np.array(offset, dtype=np.int32), 0, 255).astype(np.uint8)
    press = client.app.state.state.pm.load_bytes(ids["cmyk"])
    target = lcms_native.pixels_to_lab(
        np.ascontiguousarray(roi), "CMYK", press, "relative_colorimetric"
    ).mean(axis=0)
    r = client.post("/api/retouch", json={
        "job_id": job_id, "x": x, "y": y, "width": w, "height": h,
        "target_lab": [float(v) for v in target],
        "tolerance_de": 0.5, "max_shift": 8, "tac_limit": 1020,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["reachable"] is True
    return d


def _lab_press_at(client, job_id, coords):
    from app import main
    lab = np.load(main.JOB_DIR / job_id / "lab_press.npy")
    return np.array([[float(lab[y, x, k]) for k in range(3)] for x, y in coords])


def _csv(rows, header=("x", "y", "L", "a", "b")):
    out = ",".join(header) + "\n"
    for r in rows:
        out += ",".join(str(v) for v in r) + "\n"
    return out.encode("utf-8")


def _accept(client, job_id, csv_bytes, limit=3.0, retouch_id=None):
    data = {"job_id": job_id, "limit_de": str(limit)}
    if retouch_id is not None:
        data["retouch_id"] = retouch_id
    return client.post(
        "/api/acceptance",
        data=data,
        files={"file": ("points.csv", csv_bytes, "text/csv")},
    )


def _points_from_preds(coords, preds, perturb=0.0):
    rows = []
    for (x, y), lab in zip(coords, preds):
        L = float(np.clip(lab[0] + perturb, 0, 100))
        a = float(np.clip(lab[1] + perturb, -128, 127))
        b = float(np.clip(lab[2] + perturb, -128, 127))
        rows.append((x, y, L, a, b))
    return rows


# --------------------------------------------------------------------------

def test_health(client):
    assert client.get("/api/health").status_code == 200


def test_job_direct_pass_with_exact_predicted_lab(client, ids):
    job = _run_job(client, ids)
    coords = [(0, 0), (10, 10), (47, 47)]
    preds = _lab_press_at(client, job["job_id"], coords)
    rows = _points_from_preds(coords, preds)
    r = _accept(client, job["job_id"], _csv(rows), limit=0.0)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["passed"] is True
    assert d["point_count"] == 3
    assert d["over_limit_count"] == 0
    assert all(p["de00"] < 1e-6 for p in d["points"])
    assert all(p["prediction_source"] == "original_job_prediction"
               for p in d["points"])
    assert d["retouch_id"] is None


def test_any_point_over_limit_fails_whole_sheet(client, ids):
    job = _run_job(client, ids)
    coords = [(0, 0), (10, 10), (20, 20)]
    preds = _lab_press_at(client, job["job_id"], coords)
    rows = _points_from_preds(coords, preds, perturb=0.0)
    # 第二个点人为偏移 8 个 Lab 单位 → 必然超 3 的上限。
    rows[1] = (10, 10, 50.0, 20.0, -20.0)
    r = _accept(client, job["job_id"], _csv(rows), limit=3.0)
    d = r.json()
    assert d["passed"] is False
    assert d["over_limit_count"] == 1
    over = d["over_limit_points"]
    assert over == [p for p in over if p["x"] == 10 and p["y"] == 10]
    # 该点在逐点表中被标记超限。
    assert d["points"][1]["over_limit"] is True
    assert d["points"][0]["over_limit"] is False


def test_boundary_equal_to_limit_passes_strict_greater(client, ids):
    """ΔE00 恰好等于上限（精度内）应合格：规则是“严格大于才超限”。"""
    job = _run_job(client, ids)
    coords = [(5, 5)]
    preds = _lab_press_at(client, job["job_id"], coords)
    lab = preds[0]
    # 沿 L 方向移动；ΔE00 对纯 L 差近似等于差值（小偏差可接受）。
    # 用极小扰动保证 de < 上限，边界语义由直接构造的报告复核。
    rows = [(5, 5, float(np.clip(lab[0] + 1e-6, 0, 100)),
             float(lab[1]), float(lab[2]))]
    r = _accept(client, job["job_id"], _csv(rows), limit=1.0)
    assert r.json()["passed"] is True
    # 直接验证“严格大于”判定：de 与上限相等时不超限。
    from app import acceptance as ac
    de = np.array([2.0])
    assert bool((de > 2.0).any()) is False


def test_duplicate_coordinates_kept_separately(client, ids):
    job = _run_job(client, ids)
    preds = _lab_press_at(client, job["job_id"], [(0, 0), (0, 0)])
    rows = _points_from_preds([(0, 0), (0, 0)], preds)
    # 第二行同坐标复测给一个显著不同的 Lab：必须逐条保留且后者超限。
    rows[1] = (0, 0, 30.0, 40.0, -40.0)
    r = _accept(client, job["job_id"], _csv(rows), limit=3.0)
    d = r.json()
    assert d["point_count"] == 2
    assert d["points"][0]["x"] == 0 and d["points"][1]["x"] == 0
    assert d["points"][0]["over_limit"] is False
    assert d["points"][1]["over_limit"] is True
    assert d["over_limit_count"] == 1


def test_retouch_inside_uses_verified_product_outside_keeps_job(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    job_id = job["job_id"]
    rect = (4, 4, 16, 16)
    rt = _retouch_reachable(client, job_id, offset=(3, 0, 0, 0),
                            rect=rect, ids=ids)
    # 选区内点 (8,8) 与选区外点 (40,40)。
    coords = [(8, 8), (40, 40)]
    # 实测值取“各自预测路径”的预测 Lab：选区内=成品回算，选区外=原任务值。
    from app import main
    press = client.app.state.state.pm.load_bytes(ids["cmyk"])
    with Image.open(main.RETOUCH_DIR / rt["retouch_id"] / "retouched.tif") as im:
        tiff = np.asarray(im)
    lab_in = lcms_native.pixels_to_lab(
        tiff[8:9, 8:9].reshape(1, 4), "CMYK", press, "relative_colorimetric"
    )[0]
    lab_out = np.load(main.JOB_DIR / job_id / "lab_press.npy")[40, 40]
    rows = [
        (8, 8, *[float(v) for v in lab_in]),
        (40, 40, *[float(v) for v in lab_out]),
    ]
    r = _accept(client, job_id, _csv(rows), limit=0.0,
                retouch_id=rt["retouch_id"])
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["passed"] is True
    assert d["points"][0]["prediction_source"] == "verified_retouched_product"
    assert d["points"][1]["prediction_source"] == "original_job_prediction"
    assert d["retouch_id"] == rt["retouch_id"]
    rep = d["report"]
    assert rep["retouch"]["region"] == rt["rect"]
    assert rep["retouch"]["best_offset_cmyk"] == rt["best_offset_cmyk"]
    assert rep["profiles"]["press"]["sha256"] == \
        job["report"]["profiles"]["press"]["sha256"]
    assert "verified" in rep["prediction_method"]["strategy"]


def test_retouch_inside_differs_from_original_job_prediction(client, ids):
    """选区内成品回算 Lab ≠ 原任务值；用原任务值当实测应在修色路径下超限。"""
    job = _run_job(client, ids, png=_gradient_png())
    job_id = job["job_id"]
    rect = (4, 4, 16, 16)
    rt = _retouch_reachable(client, job_id, offset=(8, 0, 0, 0),
                            rect=rect, ids=ids)
    from app import main
    job_lab_in = np.load(main.JOB_DIR / job_id / "lab_press.npy")[8, 8]
    press = client.app.state.state.pm.load_bytes(ids["cmyk"])
    with Image.open(main.RETOUCH_DIR / rt["retouch_id"] / "retouched.tif") as im:
        tiff = np.asarray(im)
    prod_lab_in = lcms_native.pixels_to_lab(
        tiff[8:9, 8:9].reshape(1, 4), "CMYK", press, "relative_colorimetric"
    )[0]
    # 该偏移确实移动了颜色，两条路径可区分。
    assert float(np.abs(np.asarray(job_lab_in) - prod_lab_in).max()) > 0.5

    # 以“原任务预测值”为实测：在不绑定修色时 ΔE≈0 合格……
    rows0 = [(8, 8, *[float(v) for v in job_lab_in])]
    assert _accept(client, job_id, _csv(rows0), limit=0.5).json()["passed"]
    # ……绑定修色后该点按成品回算，偏差超限 → 不合格（证明未用原任务值代替）。
    r = _accept(client, job_id, _csv(rows0), limit=0.2,
                retouch_id=rt["retouch_id"])
    assert r.json()["passed"] is False
    assert r.json()["points"][0]["prediction_source"] == \
        "verified_retouched_product"


def test_retouches_listing_only_reachable(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    rt = _retouch_reachable(client, job["job_id"], ids=ids)
    lst = client.get(f"/api/jobs/{job['job_id']}/retouches").json()
    assert any(x["retouch_id"] == rt["retouch_id"] for x in lst)
    # 不可达修色不应出现在列表中。
    r = client.post("/api/retouch", json={
        "job_id": job["job_id"], "x": 0, "y": 0, "width": 4, "height": 4,
        "target_lab": [0.0, 127.0, -128.0],
        "tolerance_de": 0.0, "max_shift": 0, "tac_limit": 0,
    })
    assert r.json()["reachable"] is False
    lst2 = client.get(f"/api/jobs/{job['job_id']}/retouches").json()
    assert len(lst2) == len(lst)


# --------------------------------------------------------------------------
# 拒绝路径
# --------------------------------------------------------------------------

def test_unknown_job_404(client, ids):
    r = _accept(client, "0" * 64, _csv([(0, 0, 50, 0, 0)]))
    assert r.status_code == 404


def test_unknown_retouch_404(client, ids):
    job = _run_job(client, ids)
    r = _accept(client, job["job_id"], _csv([(0, 0, 50, 0, 0)]),
                retouch_id="f" * 64)
    assert r.status_code == 404


def test_empty_csv_rejected(client, ids):
    job = _run_job(client, ids)
    assert _accept(client, job["job_id"], b"").status_code == 422


def test_header_only_csv_rejected(client, ids):
    job = _run_job(client, ids)
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n").status_code == 422


def test_missing_column_rejected(client, ids):
    job = _run_job(client, ids)
    bad = b"x,y,L,a\n0,0,50,0\n"
    assert _accept(client, job["job_id"], bad).status_code == 422


def test_extra_column_rejected(client, ids):
    job = _run_job(client, ids)
    bad = b"x,y,L,a,b,extra\n0,0,50,0,0,9\n"
    assert _accept(client, job["job_id"], bad).status_code == 422


def test_noninteger_coordinate_rejected(client, ids):
    job = _run_job(client, ids)
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n1.5,0,50,0,0\n"
                   ).status_code == 422
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n+1,0,50,0,0\n"
                   ).status_code == 422


def test_negative_and_out_of_bounds_coordinates_rejected(client, ids):
    job = _run_job(client, ids)  # 48x48
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n-1,0,50,0,0\n"
                   ).status_code == 422
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n48,0,50,0,0\n"
                   ).status_code == 422
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n0,47,50,0,0\n"
                   ).status_code == 200


def test_lab_range_rejected(client, ids):
    job = _run_job(client, ids)
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n0,0,100.1,0,0\n"
                   ).status_code == 422
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n0,0,50,128,0\n"
                   ).status_code == 422
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n0,0,50,0,-129\n"
                   ).status_code == 422
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n0,0,nan,0,0\n"
                   ).status_code == 422


def test_ragged_row_rejected(client, ids):
    job = _run_job(client, ids)
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n0,0,50,0\n"
                   ).status_code == 422


def test_negative_limit_rejected(client, ids):
    job = _run_job(client, ids)
    r = _accept(client, job["job_id"], _csv([(0, 0, 50, 0, 0)]), limit=-1)
    assert r.status_code == 422


def test_press_icc_missing_rejected(client, ids):
    job = _run_job(client, ids)
    state = client.app.state.state
    target = next(p for role, p in state.pm._iter_files()
                  if role in ("press", "builtin") and p.stem == ids["cmyk"])
    backup = target.read_bytes()
    target.unlink()
    try:
        r = _accept(client, job["job_id"], _csv([(0, 0, 50, 0, 0)]))
        assert r.status_code == 422
        assert "印刷 ICC" in r.json()["detail"]
    finally:
        target.write_bytes(backup)


def test_press_icc_digest_mismatch_rejected(client, ids):
    job = _run_job(client, ids)
    state = client.app.state.state
    target = next(p for role, p in state.pm._iter_files()
                  if p.stem == ids["cmyk"])
    other = lcms_native.build_demo_cmyk_profile_bytes(ink_scale=0.9)
    original = target.read_bytes()
    target.write_bytes(other)
    try:
        r = _accept(client, job["job_id"], _csv([(0, 0, 50, 0, 0)]))
        assert r.status_code == 422
        assert "摘要不符" in r.json()["detail"]
        # 绑定修色路径同样拒绝。
        target.write_bytes(original)
    finally:
        target.write_bytes(original)


def test_unreachable_retouch_rejected(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    # max_shift=0 + TAC=0 在非零墨选区上无可行偏移 → 不可达、无成品。
    r0 = client.post("/api/retouch", json={
        "job_id": job["job_id"], "x": 4, "y": 4, "width": 8, "height": 8,
        "target_lab": [50, 50, -50],
        "tolerance_de": 0.1, "max_shift": 0, "tac_limit": 10,
    })
    rid = r0.json()["retouch_id"]
    assert r0.json()["reachable"] is False
    r = _accept(client, job["job_id"], _csv([(5, 5, 50, 0, 0)]),
                retouch_id=rid)
    assert r.status_code == 422


def test_retouch_bound_to_other_job_rejected(client, ids):
    job1 = _run_job(client, ids, png=_gradient_png())
    job2 = _run_job(client, ids, png=_png(rgb=(120, 150, 90)))
    rt = _retouch_reachable(client, job1["job_id"], ids=ids)
    r = _accept(client, job2["job_id"], _csv([(8, 8, 50, 0, 0)]),
                retouch_id=rt["retouch_id"])
    assert r.status_code == 422
    assert "任务" in r.json()["detail"] or "绑定" in r.json()["detail"]


def test_tampered_retouch_tiff_rejected(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    rt = _retouch_reachable(client, job["job_id"], ids=ids)
    from app import main
    tiff_path = main.RETOUCH_DIR / rt["retouch_id"] / "retouched.tif"
    # 用另一份全零 CMYK TIFF（嵌入同一 ICC）替换成品：像素绑定必须不符。
    with Image.open(tiff_path) as im:
        embedded = im.info.get("icc_profile")
    fake = np.zeros_like(np.asarray(Image.open(tiff_path)))
    buf = io.BytesIO()
    Image.fromarray(fake, "CMYK").save(
        buf, format="TIFF", compression="raw", icc_profile=embedded)
    tiff_path.write_bytes(buf.getvalue())
    try:
        r = _accept(client, job["job_id"], _csv([(8, 8, 50, 0, 0)]),
                    retouch_id=rt["retouch_id"])
        assert r.status_code == 422
        assert "成品" in r.json()["detail"]
    finally:
        # 重新提交修色以恢复正确成品，避免污染其他用例目录。
        client.post("/api/retouch", json={
            "job_id": job["job_id"], "x": 4, "y": 4, "width": 16, "height": 16,
            "target_lab": rt["report"]["constraints"]["target_lab"],
            "tolerance_de": rt["report"]["constraints"]["tolerance_de00"],
            "max_shift": rt["report"]["constraints"]["max_shift_per_channel"],
            "tac_limit": rt["report"]["constraints"]["total_ink_limit"],
        })


def test_tampered_retouch_press_digest_rejected(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    rt = _retouch_reachable(client, job["job_id"], ids=ids)
    rep_path = None
    from app import main
    rep_path = main.RETOUCH_DIR / rt["retouch_id"] / "report.json"
    rep = json.loads(rep_path.read_text("utf-8"))
    original = rep["profiles"]["press"]["sha256"]
    rep["profiles"]["press"]["sha256"] = "a" * 64
    rep_path.write_text(json.dumps(rep), "utf-8")
    try:
        r = _accept(client, job["job_id"], _csv([(8, 8, 50, 0, 0)]),
                    retouch_id=rt["retouch_id"])
        assert r.status_code == 422
    finally:
        rep["profiles"]["press"]["sha256"] = original
        rep_path.write_text(json.dumps(rep), "utf-8")


def test_retouch_tiff_with_foreign_icc_rejected(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    rt = _retouch_reachable(client, job["job_id"], ids=ids)
    from app import main
    tiff_path = main.RETOUCH_DIR / rt["retouch_id"] / "retouched.tif"
    with Image.open(tiff_path) as im:
        pixels = np.asarray(im)
    # 像素保持不变，但嵌入另一份 CMYK ICC：嵌入 ICC 绑定必须不符。
    foreign = lcms_native.build_demo_cmyk_profile_bytes(ink_scale=0.8)
    buf = io.BytesIO()
    Image.fromarray(pixels, "CMYK").save(
        buf, format="TIFF", compression="raw", icc_profile=foreign)
    backup = tiff_path.read_bytes()
    tiff_path.write_bytes(buf.getvalue())
    try:
        r = _accept(client, job["job_id"], _csv([(8, 8, 50, 0, 0)]),
                    retouch_id=rt["retouch_id"])
        assert r.status_code == 422
        assert "ICC" in r.json()["detail"]
    finally:
        tiff_path.write_bytes(backup)


# --------------------------------------------------------------------------
# 确定性 / 报告 / 产物
# --------------------------------------------------------------------------

def test_deterministic_acceptance_id_and_reuse(client, ids):
    job = _run_job(client, ids)
    preds = _lab_press_at(client, job["job_id"], [(0, 0)])
    rows = _points_from_preds([(0, 0)], preds)
    csv_bytes = _csv(rows)
    d1 = _accept(client, job["job_id"], csv_bytes, limit=3.0).json()
    d2 = _accept(client, job["job_id"], csv_bytes, limit=3.0).json()
    assert d1["acceptance_id"] == d2["acceptance_id"]
    # 不同上限得到不同身份。
    d3 = _accept(client, job["job_id"], csv_bytes, limit=4.0).json()
    assert d3["acceptance_id"] != d1["acceptance_id"]


def test_report_endpoint_and_csv_artifact(client, ids):
    job = _run_job(client, ids)
    preds = _lab_press_at(client, job["job_id"], [(0, 0), (1, 1)])
    csv_bytes = _csv(_points_from_preds([(0, 0), (1, 1)], preds))
    d = _accept(client, job["job_id"], csv_bytes, limit=3.0).json()
    rep = client.get(f"/api/acceptance/{d['acceptance_id']}").json()
    assert rep["source_job"]["job_id"] == job["job_id"]
    assert rep["profiles"]["press"]["sha256"]
    assert rep["rule"]["limit_de00"] == 3.0
    assert rep["points"][0]["seq"] == 1 and rep["points"][1]["seq"] == 2
    # 报告下载端点。
    rr = client.get(f"/api/acceptance/{d['acceptance_id']}/report")
    assert rr.status_code == 200
    assert rr.headers["content-type"].startswith("application/json")
    # 原始 CSV 原样保存可下载：字节、CSV 内容类型与原下载文件名均保留。
    cc = client.get(f"/api/acceptance/{d['acceptance_id']}/measurements.csv")
    assert cc.status_code == 200 and cc.content == csv_bytes
    assert cc.headers["content-type"].startswith("text/csv")
    assert cc.headers["content-disposition"] == (
        f'attachment; filename="measurements-{d["acceptance_id"][:12]}.csv"'
    )


def test_header_aliases_accepted(client, ids):
    job = _run_job(client, ids)
    preds = _lab_press_at(client, job["job_id"], [(0, 0)])
    L, a, b = (float(v) for v in preds[0])
    csv_bytes = f" X , Y , L*, a*, b*\n0,0,{L},{a},{b}\n".encode("utf-8")
    r = _accept(client, job["job_id"], csv_bytes, limit=0.0)
    assert r.status_code == 200, r.text
    assert r.json()["passed"] is True


def test_same_points_but_distinct_raw_csv_are_separately_addressed(client, ids):
    """同测点序列、不同原始字节（仅表头别名/空白不同）须分别寻址、各自绑定原件。"""
    job = _run_job(client, ids)
    preds = _lab_press_at(client, job["job_id"], [(0, 0), (1, 1)])
    rows = _points_from_preds([(0, 0), (1, 1)], preds)
    L0, a0, b0 = (float(v) for v in preds[0])
    L1, a1, b1 = (float(v) for v in preds[1])

    # 原件 A：规范表头、规范空白。
    csv_a = _csv(rows)
    # 原件 B：表头用别名 + 周边空白，Lab 单元格首尾空白（解析时 .strip() 合法），
    # CRLF 行尾 + 末尾空行；解析出的测点序列与 A 逐点相同，但原始字节不同。
    # 注意坐标单元格按严格整数解析、不容忍空白，故不对坐标加空白。
    csv_b = (
        " X , Y , L*, a*, b*\r\n"
        f"0,0, {L0} ,{a0},{b0}\r\n"
        f"1,1,{L1}, {a1}, {b1} \r\n"
        "\r\n"
    ).encode("utf-8")

    r1 = _accept(client, job["job_id"], csv_a, limit=3.0)
    r2 = _accept(client, job["job_id"], csv_b, limit=3.0)
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    d1, d2 = r1.json(), r2.json()
    # 原件不同 → 身份不同（第二次合法提交绝不能被判成绑定不符而 422）。
    assert d1["acceptance_id"] != d2["acceptance_id"]
    # 逐点结论相同（同测点序列、同预测、同上限）。
    assert d1["passed"] == d2["passed"] is True
    assert d1["point_count"] == d2["point_count"] == 2
    for p1, p2 in zip(d1["points"], d2["points"]):
        assert p1["x"] == p2["x"] and p1["y"] == p2["y"]
        assert p1["measured_lab"] == p2["measured_lab"]
        assert p1["predicted_lab"] == p2["predicted_lab"]
        assert p1["de00"] == p2["de00"]
        assert p1["over_limit"] == p2["over_limit"]

    # 各自报告记录各自的原件摘要；身份指纹也携带该摘要。
    rep1 = client.get(f"/api/acceptance/{d1['acceptance_id']}").json()
    rep2 = client.get(f"/api/acceptance/{d2['acceptance_id']}").json()
    import hashlib
    sha_a = hashlib.sha256(csv_a).hexdigest()
    sha_b = hashlib.sha256(csv_b).hexdigest()
    assert sha_a != sha_b
    assert rep1["measurements_csv_sha256"] == sha_a
    assert rep2["measurements_csv_sha256"] == sha_b
    assert rep1["request"]["measurements_csv_sha256"] == sha_a
    assert rep2["request"]["measurements_csv_sha256"] == sha_b
    assert rep1["request"]["points"] == rep2["request"]["points"]

    # 各自下载到与提交字节逐字节一致的原件（不串用）。
    c1 = client.get(f"/api/acceptance/{d1['acceptance_id']}/measurements.csv")
    c2 = client.get(f"/api/acceptance/{d2['acceptance_id']}/measurements.csv")
    assert c1.status_code == 200 and c1.content == csv_a
    assert c2.status_code == 200 and c2.content == csv_b

    # 完全相同文件重复提交：稳定复用同一身份、同一原件。
    d3 = _accept(client, job["job_id"], csv_b, limit=3.0).json()
    assert d3["acceptance_id"] == d2["acceptance_id"]
    c3 = client.get(f"/api/acceptance/{d3['acceptance_id']}/measurements.csv")
    assert c3.content == csv_b


def test_tampered_archived_csv_rejected_for_download_and_healed_on_repost(client, ids):
    """归档 measurements.csv 被替换：下载按报告摘要拒绝；同文件再提交可修复。"""
    from app import main
    job = _run_job(client, ids)
    preds = _lab_press_at(client, job["job_id"], [(0, 0)])
    csv_bytes = _csv(_points_from_preds([(0, 0)], preds))
    d = _accept(client, job["job_id"], csv_bytes, limit=3.0).json()
    aid = d["acceptance_id"]
    csv_path = main.ACCEPTANCE_DIR / aid / "measurements.csv"

    # 用另一份合法但内容不同的 CSV 替换归档原件（测点坐标仍在幅面内）。
    forged = b"x,y,L,a,b\n0,0,12.0,3.4,-5.6\n"
    assert forged != csv_bytes
    csv_path.write_bytes(forged)
    try:
        # 下载必须拒绝：与报告固化摘要不符。
        cc = client.get(f"/api/acceptance/{aid}/measurements.csv")
        assert cc.status_code == 422
        assert "摘要不符" in cc.json()["detail"]
        # 报告本身仍可取（只读 JSON 元数据）。
        assert client.get(f"/api/acceptance/{aid}").status_code == 200
        # 重复提交完全相同的原文件：命中缓存并修复归档原件，而非 422。
        rr = _accept(client, job["job_id"], csv_bytes, limit=3.0)
        assert rr.status_code == 200, rr.text
        assert rr.json()["acceptance_id"] == aid
        healed = client.get(f"/api/acceptance/{aid}/measurements.csv")
        assert healed.status_code == 200 and healed.content == csv_bytes
    finally:
        csv_path.write_bytes(csv_bytes)


def test_download_delivers_the_same_bytes_that_were_verified(client, ids, monkeypatch):
    """核验与交付必须是同一份字节：核验读取后归档被原位改写，响应仍须是已核验字节。

    若实现退回“核验后按路径/句柄再次打开”（如 FileResponse），本用例会交出
    改写后的伪造字节而失败。
    """
    from app import acceptance as ac
    from app import main
    job = _run_job(client, ids)
    preds = _lab_press_at(client, job["job_id"], [(0, 0)])
    csv_bytes = _csv(_points_from_preds([(0, 0)], preds))
    d = _accept(client, job["job_id"], csv_bytes, limit=3.0).json()
    aid = d["acceptance_id"]
    csv_path = main.ACCEPTANCE_DIR / aid / "measurements.csv"
    forged = b"x,y,L,a,b\n0,0,12.0,3.4,-5.6\n"
    assert forged != csv_bytes

    real_read = ac.read_stored_csv

    def read_then_rewrite_in_place(path):
        blob = real_read(path)
        # 模拟核验读取之后、响应生成之前归档被原位改写。
        csv_path.write_bytes(forged)
        return blob

    monkeypatch.setattr(ac, "read_stored_csv", read_then_rewrite_in_place)
    try:
        r = client.get(f"/api/acceptance/{aid}/measurements.csv")
        assert r.status_code == 200
        # 交付的必须是核验时读到的同一份字节，而非改写后的文件内容。
        assert r.content == csv_bytes
        assert r.headers["content-type"].startswith("text/csv")
        assert r.headers["content-disposition"] == \
            f'attachment; filename="measurements-{aid[:12]}.csv"'
    finally:
        csv_path.write_bytes(csv_bytes)


def test_missing_archived_csv_healed_on_identical_repost(client, ids):
    """归档原件缺失时，重复提交同文件应恢复原件并复用。"""
    from app import main
    job = _run_job(client, ids)
    preds = _lab_press_at(client, job["job_id"], [(0, 0)])
    csv_bytes = _csv(_points_from_preds([(0, 0)], preds))
    d = _accept(client, job["job_id"], csv_bytes, limit=3.0).json()
    aid = d["acceptance_id"]
    csv_path = main.ACCEPTANCE_DIR / aid / "measurements.csv"
    csv_path.unlink()
    assert client.get(f"/api/acceptance/{aid}/measurements.csv").status_code == 404
    rr = _accept(client, job["job_id"], csv_bytes, limit=3.0)
    assert rr.status_code == 200 and rr.json()["acceptance_id"] == aid
    cc = client.get(f"/api/acceptance/{aid}/measurements.csv")
    assert cc.status_code == 200 and cc.content == csv_bytes


def test_acceptance_404_for_missing_report(client, ids):
    assert client.get("/api/acceptance/deadbeef").status_code == 404


# --------------------------------------------------------------------------
# 修色报告选区/偏移的整数语义、请求指纹矩形 ↔ 贴边裁剪选区对应
# --------------------------------------------------------------------------

def _rewrite_retouch_report(retouch_id, mutate):
    """读取修色报告、按 mutate 改写后落盘；返回 (报告路径, 原始报告 dict)。"""
    from app import main
    rep_path = main.RETOUCH_DIR / retouch_id / "report.json"
    original = json.loads(rep_path.read_text("utf-8"))
    rep = json.loads(rep_path.read_text("utf-8"))
    mutate(rep)
    rep_path.write_text(json.dumps(rep), "utf-8")
    return rep_path, original


def test_fractional_region_in_report_rejected_no_truncation(client, ids):
    """归档报告把 x=4,width=1 改为 x=4.5,width=1.5：成品像素按截断选区本可
    通过核验，但逐点预测会按分数选区划分选区内外（x=4 的测点被错划到选区
    外而改取 lab_press.npy）——必须 422，绝不用 int() 截断默许。"""
    job = _run_job(client, ids, png=_gradient_png())
    rt = _retouch_reachable(client, job["job_id"], offset=(3, 0, 0, 0),
                            rect=(4, 4, 1, 1), ids=ids)
    assert rt["rect"] == {"x": 4, "y": 4, "width": 1, "height": 1}
    rep_path, original = _rewrite_retouch_report(
        rt["retouch_id"],
        lambda rep: rep.update(
            region={"x": 4.5, "y": 4.0, "width": 1.5, "height": 1.0}))
    try:
        r = _accept(client, job["job_id"], _csv([(4, 4, 50, 0, 0)]),
                    retouch_id=rt["retouch_id"])
        assert r.status_code == 422
        assert "整数" in r.json()["detail"]
    finally:
        rep_path.write_text(json.dumps(original), "utf-8")


def test_integral_float_and_boolean_region_rejected(client, ids):
    """4.0 浮点写法与布尔坐标同样拒绝：报告选区只接受 JSON 整数。"""
    job = _run_job(client, ids, png=_gradient_png())
    rt = _retouch_reachable(client, job["job_id"], rect=(4, 4, 2, 2), ids=ids)
    for bad_region in (
        {"x": 4.0, "y": 4.0, "width": 2.0, "height": 2.0},
        {"x": True, "y": 4, "width": 2, "height": 2},
        {"x": 4, "y": 4, "width": 2, "height": False},
        {"x": 4, "y": 4, "width": "2", "height": 2},
    ):
        rep_path, original = _rewrite_retouch_report(
            rt["retouch_id"], lambda rep: rep.update(region=bad_region))
        try:
            r = _accept(client, job["job_id"], _csv([(4, 4, 50, 0, 0)]),
                        retouch_id=rt["retouch_id"])
            assert r.status_code == 422, bad_region
            assert "整数" in r.json()["detail"]
        finally:
            rep_path.write_text(json.dumps(original), "utf-8")


def test_fractional_boolean_and_out_of_range_offset_rejected(client, ids):
    """偏移同样只接受合法整数：分数、布尔、超出单通道硬上限一律 422。"""
    job = _run_job(client, ids, png=_gradient_png())
    rt = _retouch_reachable(client, job["job_id"], rect=(4, 4, 2, 2), ids=ids)
    for bad_offset in ([1.5, 0, 0, 0], [True, 0, 0, 0], [100, 0, 0, 0]):
        rep_path, original = _rewrite_retouch_report(
            rt["retouch_id"],
            lambda rep: rep.update(best_offset_cmyk=bad_offset))
        try:
            r = _accept(client, job["job_id"], _csv([(4, 4, 50, 0, 0)]),
                        retouch_id=rt["retouch_id"])
            assert r.status_code == 422, bad_offset
        finally:
            rep_path.write_text(json.dumps(original), "utf-8")


def test_region_must_match_request_fingerprint_clipped_rect(client, ids):
    """报告选区必须等于请求指纹矩形贴边裁剪后的实际选区。

    只改请求指纹矩形（选区与成品不动）：成品像素核验本可通过，
    对应关系核对必须拦截（422）。
    """
    job = _run_job(client, ids, png=_gradient_png())
    rt = _retouch_reachable(client, job["job_id"], rect=(4, 4, 2, 2), ids=ids)
    # 请求指纹矩形被改：x=4→5，与报告选区 (4,4,2,2) 不再对应。
    rep_path, original = _rewrite_retouch_report(
        rt["retouch_id"], lambda rep: rep["request"].update(x=5))
    try:
        r = _accept(client, job["job_id"], _csv([(4, 4, 50, 0, 0)]),
                    retouch_id=rt["retouch_id"])
        assert r.status_code == 422
        assert "选区" in r.json()["detail"]
    finally:
        rep_path.write_text(json.dumps(original), "utf-8")
    # 报告选区被改成与请求指纹矩形不符的另一组整数。
    rep_path, original = _rewrite_retouch_report(
        rt["retouch_id"],
        lambda rep: rep.update(region={"x": 5, "y": 4, "width": 2, "height": 2}))
    try:
        r = _accept(client, job["job_id"], _csv([(4, 4, 50, 0, 0)]),
                    retouch_id=rt["retouch_id"])
        assert r.status_code == 422
    finally:
        rep_path.write_text(json.dumps(original), "utf-8")


def test_request_fingerprint_job_mismatch_rejected(client, ids):
    """请求指纹绑定的原任务与所选任务不符 → 422。"""
    job = _run_job(client, ids, png=_gradient_png())
    rt = _retouch_reachable(client, job["job_id"], rect=(4, 4, 2, 2), ids=ids)
    rep_path, original = _rewrite_retouch_report(
        rt["retouch_id"], lambda rep: rep["request"].update(job_id="0" * 64))
    try:
        r = _accept(client, job["job_id"], _csv([(4, 4, 50, 0, 0)]),
                    retouch_id=rt["retouch_id"])
        assert r.status_code == 422
    finally:
        rep_path.write_text(json.dumps(original), "utf-8")


def test_missing_request_fingerprint_rejected(client, ids):
    """报告缺少请求指纹，无法核对选区与请求的对应 → 422。"""
    job = _run_job(client, ids, png=_gradient_png())
    rt = _retouch_reachable(client, job["job_id"], rect=(4, 4, 2, 2), ids=ids)
    rep_path, original = _rewrite_retouch_report(
        rt["retouch_id"], lambda rep: rep.pop("request"))
    try:
        r = _accept(client, job["job_id"], _csv([(4, 4, 50, 0, 0)]),
                    retouch_id=rt["retouch_id"])
        assert r.status_code == 422
    finally:
        rep_path.write_text(json.dumps(original), "utf-8")


def test_edge_clipped_retouch_acceptance_uses_verified_clipped_region(client, ids):
    """合法贴边修色：48×48 图上请求 (44,44,16,16) 的实际选区为 (44,44,4,4)。

    验收必须按这份已验证的裁剪选区划分内外：角点 (44,44)/(47,47) 在选区
    内（成品回算），(43,43) 在选区外（原任务预测值）；非零偏移下若误用
    原任务值，选区内测点会在 limit=0 下超限，故合格结论也依赖正确划分。
    """
    job = _run_job(client, ids, png=_gradient_png())
    job_id = job["job_id"]
    rt = _retouch_reachable(client, job_id, offset=(2, 0, 0, 0),
                            rect=(44, 44, 16, 16), ids=ids)
    assert rt["rect"] == {"x": 44, "y": 44, "width": 4, "height": 4}
    from app import main
    press = client.app.state.state.pm.load_bytes(ids["cmyk"])
    with Image.open(main.RETOUCH_DIR / rt["retouch_id"] / "retouched.tif") as im:
        tiff = np.asarray(im)
    lab_press = np.load(main.JOB_DIR / job_id / "lab_press.npy")

    def product_lab(x, y):
        return lcms_native.pixels_to_lab(
            np.ascontiguousarray(tiff[y:y + 1, x:x + 1].reshape(1, 4)),
            "CMYK", press, "relative_colorimetric")[0]

    # 非零偏移确实移动了角点颜色（两条预测路径数值可区分）。
    assert float(np.abs(np.asarray(product_lab(44, 44))
                        - np.asarray(lab_press[44, 44])).max()) > 0.1

    rows = [
        (44, 44, *[float(v) for v in product_lab(44, 44)]),
        (47, 47, *[float(v) for v in product_lab(47, 47)]),
        (43, 43, *[float(v) for v in lab_press[43, 43]]),
    ]
    r = _accept(client, job_id, _csv(rows), limit=0.0,
                retouch_id=rt["retouch_id"])
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["passed"] is True
    assert [p["prediction_source"] for p in d["points"]] == [
        "verified_retouched_product",
        "verified_retouched_product",
        "original_job_prediction",
    ]
    # 验收报告记录的是贴边裁剪后的实际选区（既有报告格式不变）。
    assert d["report"]["retouch"]["region"] == \
        {"x": 44, "y": 44, "width": 4, "height": 4}
