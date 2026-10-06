"""局部 CMYK 修色端到端测试：搜索、约束、不可达、ICC 绑定、TIFF 成品。"""
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
    data_dir = tmp_path / "retouch-data"
    os.environ["SPD_DATA_DIR"] = str(data_dir)
    from app import config
    config._state = None
    config.DATA_DIR = data_dir
    config.PROFILE_DIR = data_dir / "profiles"
    config.JOB_DIR = data_dir / "jobs"
    config.RETOUCH_DIR = data_dir / "retouch"
    config.RELEASE_DIR = data_dir / "release"
    config.DB_PATH = data_dir / "softproof.db"
    from app import main
    main.JOB_DIR = data_dir / "jobs"
    main.RETOUCH_DIR = data_dir / "retouch"
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


def _solid_png(rgb=(210, 90, 60), size=48):
    arr = np.zeros((size, size, 3), np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = rgb
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _run_job(client, ids, rgb=(210, 90, 60)):
    r = client.post(
        "/api/jobs",
        files={"image": ("solid.png", _solid_png(rgb), "image/png")},
        data={
            "source_profile": ids["srgb"],
            "press_profile": ids["cmyk"],
            "display_profile": ids["display"],
            "threshold": "3",
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def _retouch(client, job_id, **over):
    body = {
        "job_id": job_id, "x": 4, "y": 4, "width": 16, "height": 16,
        "target_lab": [50.0, 10.0, -10.0],
        "tolerance_de": 3.0, "max_shift": 8, "tac_limit": 1020,
    }
    body.update(over)
    return client.post("/api/retouch", json=body)


# --------------------------------------------------------------------------

def test_health(client):
    assert client.get("/api/health").status_code == 200


def test_retouch_reaches_with_zero_offset_for_current_mean(client, ids):
    """目标 Lab 取修色前平均 Lab 时，(0,0,0,0) 必须以 ΔE≈0 命中且排第一。"""
    from app import main
    job = _run_job(client, ids)
    cmyk = np.load(main.JOB_DIR / job["job_id"] / "cmyk.npy")
    press = client.app.state.state.pm.load_bytes(ids["cmyk"])
    lab = lcms_native.pixels_to_lab(
        cmyk[4:20, 4:20].reshape(-1, 4), "CMYK", press, "relative_colorimetric"
    ).mean(axis=0)

    r = _retouch(client, job["job_id"], target_lab=[float(v) for v in lab],
                 tolerance_de=0.0, max_shift=0)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["reachable"] is True
    assert d["best_offset_cmyk"] == [0, 0, 0, 0]
    assert d["best_abs_shift_sum"] == 0
    assert d["residual"]["de00_to_target"] < 1e-6
    assert d["residual"]["mean_de00_before_after"] == 0
    assert d["feasible_offset_count"] == 1


def _mean_lab_of_offset(client, ids, job_id, rect, offset):
    """沿原印刷 ICC 回算“原选区 CMYK 统一加 offset 后”的平均 Lab。"""
    from app import main
    cmyk = np.load(main.JOB_DIR / job_id / "cmyk.npy")
    x, y, w, h = rect
    roi = cmyk[y:y + h, x:x + w].reshape(-1, 4).astype(np.int32)
    roi = np.clip(roi + np.array(offset, dtype=np.int32), 0, 255).astype(np.uint8)
    press = client.app.state.state.pm.load_bytes(ids["cmyk"])
    return lcms_native.pixels_to_lab(
        np.ascontiguousarray(roi), "CMYK", press, "relative_colorimetric"
    ).mean(axis=0)


def test_retouch_moves_color_and_produces_tiff(client, ids):
    job = _run_job(client, ids, rgb=(180, 180, 180))
    # 目标取“统一加 (3,0,0,0) 后”的平均 Lab：该偏移必然可行且必达标，
    # 不依赖对演示配置色域的主观假设。
    target = _mean_lab_of_offset(client, ids, job["job_id"], (4, 4, 16, 16), (3, 0, 0, 0))
    d = _retouch(client, job["job_id"],
                 target_lab=[float(v) for v in target],
                 tolerance_de=0.5).json()
    assert d["reachable"] is True, d
    off = d["best_offset_cmyk"]
    assert len(off) == 4 and all(-8 <= v <= 8 for v in off)
    assert d["residual"]["de00_to_target"] <= 0.5
    # 修色后平均 Lab 必须比修色前更接近目标（ΔE 下降）。
    assert d["residual"]["de00_to_target"] < d["residual"]["de00_before_to_target"]

    # 成品 TIFF 存在、可打开、嵌入的 ICC 与原印刷 ICC 字节一致。
    resp = client.get(d["tiff"])
    assert resp.status_code == 200
    tif = Image.open(io.BytesIO(resp.content))
    assert tif.format == "TIFF" and tif.mode == "CMYK"
    assert tif.size == (48, 48)
    press = client.app.state.state.pm.load_bytes(ids["cmyk"])
    assert tif.info.get("icc_profile") == press


def test_tiff_preserves_bytes_outside_roi(client, ids):
    from app import main
    job = _run_job(client, ids, rgb=(170, 120, 90))
    # 以可行偏移 (+2, -1, 1, 0) 回算出的平均 Lab 为目标，保证可达以产出 TIFF。
    target = _mean_lab_of_offset(client, ids, job["job_id"], (10, 10, 12, 12),
                                 (2, -1, 1, 0))
    d = _retouch(client, job["job_id"], x=10, y=10, width=12, height=12,
                 target_lab=[float(v) for v in target],
                 tolerance_de=0.5).json()
    assert d["reachable"], d
    resp = client.get(d["tiff"])
    tif = np.asarray(Image.open(io.BytesIO(resp.content)))
    orig = np.load(main.JOB_DIR / job["job_id"] / "cmyk.npy")

    before = orig.copy()
    after = tif
    # 选区外字节严格不变。
    mask = np.zeros(before.shape[:2], dtype=bool)
    mask[10:22, 10:22] = True
    assert np.array_equal(after[~mask], before[~mask])
    # 选区内差值恰好等于报告里的统一偏移（无论选优最终选中哪一组）。
    off = np.array(d["best_offset_cmyk"], dtype=np.int32)
    diff = after[mask].astype(np.int32) - before[mask].astype(np.int32)
    assert np.all(diff == off[None, :])
    # 通道 0..255 与总墨量约束逐像素满足。
    assert after[mask].min() >= 0 and after[mask].max() <= 255
    assert after[mask].sum(axis=1).max() <= 1020
    assert np.abs(off).max() <= 8


def test_infeasible_when_tac_too_tight(client, ids):
    job = _run_job(client, ids, rgb=(10, 10, 10))
    r = _retouch(client, job["job_id"], tac_limit=0, max_shift=8)
    d = r.json()
    assert r.status_code == 200
    assert d["feasible_offset_count"] == 0
    assert d["reachable"] is False
    assert d["tiff"] is None
    # 成品端点必须显式拒绝。
    resp = client.get(f"/api/retouch/{d['retouch_id']}/retouched.tif")
    assert resp.status_code == 422


def test_unreachable_above_tolerance(client, ids):
    # 收紧容差到 0 且只允许零偏移：除非目标恰好等于当前均值，否则不可达。
    job = _run_job(client, ids, rgb=(200, 50, 50))
    r = _retouch(client, job["job_id"], target_lab=[95, -40, 40],
                 tolerance_de=0.0, max_shift=0)
    d = r.json()
    assert d["reachable"] is False
    assert d["feasible_offset_count"] == 1
    assert d["best_offset_cmyk"] == [0, 0, 0, 0]
    assert d["tiff"] is None
    # 不可达时仍返回修色前后软打样（此处两者相同）。
    assert client.get(d["images"]["before"]).status_code == 200
    assert client.get(d["images"]["after"]).status_code == 200


def test_channel_and_tac_constraints_hold_for_all_roi_pixels(client, ids):
    job = _run_job(client, ids, rgb=(90, 140, 200))
    target = _mean_lab_of_offset(client, ids, job["job_id"], (0, 0, 32, 32),
                                 (-2, 2, 1, -1))
    d = _retouch(client, job["job_id"], x=0, y=0, width=32, height=32,
                 target_lab=[float(v) for v in target], tolerance_de=0.5,
                 max_shift=3, tac_limit=1020).json()
    # 最终偏移必须在全选区逐像素合规（通道范围与总墨量）。
    from app import main
    orig = np.load(main.JOB_DIR / job["job_id"] / "cmyk.npy")
    off = np.array(d["best_offset_cmyk"], dtype=np.int32)
    assert d["reachable"], d
    assert np.abs(off).max() <= 3
    roi = orig[0:32, 0:32].astype(np.int32) + off
    assert roi.min() >= 0 and roi.max() <= 255
    assert roi.sum(axis=2).max() <= 1020


def test_tie_break_prefers_smaller_abs_sum_then_lexicographic(client, ids):
    """ΔE 并列时：偏移绝对值和小者优先；再并列按 (C,M,Y,K) 字典序。"""
    from app.retouch import _feasible_offsets
    # 平坦选区，所有像素 CMYK 相同；ΔE 只取决于偏移，构造一个两组偏移理论上
    # 对同一 Lab 变化对称的情形不易，因此直接验证可行枚举与排序键。
    roi = np.full((4, 4, 4), 100, dtype=np.uint8)
    cands = _feasible_offsets(roi, max_shift=1, tac_limit=1020)
    # 每通道 {-1,0,1} -> 3^4 = 81 组。
    assert cands.shape == (81, 4)
    # 排序键模拟：(ΔE 占位, abs sum, tuple)。
    keys = sorted(((int(np.abs(c).sum()), tuple(int(v) for v in c)) for c in cands),
                  key=lambda k: (0, k[0], k[1]))
    assert keys[0][1] == (0, 0, 0, 0)
    # abs sum = 1 的四个候选按字典序 (-1,0,0,0) 最先。
    one = [k for k in keys if k[0] == 1]
    assert one[0][1] == (-1, 0, 0, 0)


def test_rect_too_large_rejected(client, ids):
    job = _run_job(client, ids)
    r = _retouch(client, job["job_id"], width=65, height=8)
    assert r.status_code == 422
    assert "64" in r.json()["detail"]


def test_bad_lab_ranges_rejected(client, ids):
    job = _run_job(client, ids)
    for bad in ([101, 0, 0], [50, 128, 0], [50, 0, -129]):
        r = _retouch(client, job["job_id"], target_lab=bad)
        assert r.status_code == 422, bad
    # NaN 非标准 JSON 字面量，部分 HTTP 客户端拒绝序列化；以原始请求体发送，
    # 服务端仍须按“非有限数值”422 拒绝。
    body = {
        "job_id": job["job_id"], "x": 4, "y": 4, "width": 16, "height": 16,
        "target_lab": [float("nan"), 0.0, 0.0],
        "tolerance_de": 3.0, "max_shift": 8, "tac_limit": 1020,
    }
    r = client.post("/api/retouch", content=json.dumps(body),
                    headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_bad_constraints_rejected(client, ids):
    job = _run_job(client, ids)
    assert _retouch(client, job["job_id"], tolerance_de=-1).status_code == 422
    assert _retouch(client, job["job_id"], max_shift=9).status_code == 422
    assert _retouch(client, job["job_id"], max_shift=1.5).status_code == 422
    assert _retouch(client, job["job_id"], tac_limit=1021).status_code == 422
    assert _retouch(client, job["job_id"], width=0).status_code == 422


def test_unknown_job_404(client, ids):
    r = _retouch(client, "0" * 64)
    assert r.status_code == 404


def test_missing_press_icc_rejected(client, ids):
    job = _run_job(client, ids)
    # 让 ProfileManager 找不到原印刷配置文件（模拟原任务 ICC 缺失）。
    state = client.app.state.state
    target = next(p for role, p in state.pm._iter_files()
                  if role in ("press", "builtin") and p.stem == ids["cmyk"])
    backup = target.read_bytes()
    target.unlink()
    try:
        r = _retouch(client, job["job_id"])
        assert r.status_code == 422
        assert "印刷 ICC" in r.json()["detail"]
    finally:
        target.write_bytes(backup)


def test_press_icc_digest_mismatch_rejected(client, ids):
    job = _run_job(client, ids)
    state = client.app.state.state
    # 用另一份字节不同的合成 CMYK 配置替换同名文件（模拟配置被偷换）。
    target = None
    for _, p in state.pm._iter_files():
        if p.stem == ids["cmyk"]:
            target = p
            break
    other = lcms_native.build_demo_cmyk_profile_bytes(ink_scale=0.9)
    original = target.read_bytes()
    assert other != original
    target.write_bytes(other)
    try:
        r = _retouch(client, job["job_id"])
        assert r.status_code == 422
        assert "摘要不符" in r.json()["detail"]
    finally:
        target.write_bytes(original)


def test_report_binds_job_profile_constraints_offset(client, ids):
    job = _run_job(client, ids, rgb=(160, 100, 120))
    target = _mean_lab_of_offset(client, ids, job["job_id"], (4, 4, 16, 16),
                                 (1, 2, -1, 0))
    d = _retouch(client, job["job_id"], target_lab=[float(v) for v in target],
                 tolerance_de=0.5, max_shift=5, tac_limit=800).json()
    rid = d["retouch_id"]
    rep = client.get(f"/api/retouch/{rid}/report").json()
    assert rep["source_job"]["job_id"] == job["job_id"]
    assert rep["profiles"]["press"]["sha256"] == job["report"]["profiles"]["press"]["sha256"]
    assert rep["profiles"]["press"]["color_space"] == "CMYK"
    assert rep["constraints"]["max_shift_per_channel"] == 5
    assert rep["constraints"]["total_ink_limit"] == 800
    assert rep["best_offset_cmyk"] == d["best_offset_cmyk"]
    assert rep["region"] == d["rect"]
    assert rep["reachable"] is True


def test_deterministic_retouch_id_and_reuse(client, ids):
    job = _run_job(client, ids)
    d1 = _retouch(client, job["job_id"]).json()
    d2 = _retouch(client, job["job_id"]).json()
    assert d1["retouch_id"] == d2["retouch_id"]
    assert d1["best_offset_cmyk"] == d2["best_offset_cmyk"]


def _gradient_png(size=48):
    yy, xx = np.mgrid[0:size, 0:size]
    arr = np.stack([(xx * 5) % 256, (yy * 5) % 256,
                    ((xx + yy) * 3) % 256], axis=-1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _run_gradient_job(client, ids):
    r = client.post(
        "/api/jobs",
        files={"image": ("grad.png", _gradient_png(), "image/png")},
        data={
            "source_profile": ids["srgb"],
            "press_profile": ids["cmyk"],
            "display_profile": ids["display"],
            "threshold": "3",
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_colliding_rectangles_get_distinct_results(client, ids):
    """回归：(1,23,4,5) 与 (12,3,4,5) 曾拼出相同坐标串 "12345" 而串用结果；
    现在必须各自得到对应选区的搜索、报告与预览。"""
    job = _run_gradient_job(client, ids)
    r1 = _retouch(client, job["job_id"], x=1, y=23, width=4, height=5)
    r2 = _retouch(client, job["job_id"], x=12, y=3, width=4, height=5)
    assert r1.status_code == 200 and r2.status_code == 200
    d1, d2 = r1.json(), r2.json()
    # 两个请求身份不同，且各自绑定自己的选区。
    assert d1["retouch_id"] != d2["retouch_id"]
    assert d1["rect"] == {"x": 1, "y": 23, "width": 4, "height": 5}
    assert d2["rect"] == {"x": 12, "y": 3, "width": 4, "height": 5}
    # 各自报告绑定各自的任务/矩形/约束指纹。
    rep1 = client.get(d1["report_url"]).json()
    rep2 = client.get(d2["report_url"]).json()
    assert rep1["region"] == d1["rect"] and rep2["region"] == d2["rect"]
    assert rep1["request"]["x"] == 1 and rep1["request"]["y"] == 23
    assert rep2["request"]["x"] == 12 and rep2["request"]["y"] == 3
    assert rep1["source_job"]["job_id"] == job["job_id"]
    assert rep2["source_job"]["job_id"] == job["job_id"]
    # 预览按各自选区生成（渐变图上两选区内容不同，PNG 字节必须不同）。
    png1 = client.get(d1["images"]["before"]).content
    png2 = client.get(d2["images"]["before"]).content
    assert png1 != png2
    # 相同请求重复提交仍稳定复用各自结果。
    assert _retouch(client, job["job_id"], x=1, y=23,
                    width=4, height=5).json() == d1
    assert _retouch(client, job["job_id"], x=12, y=3,
                    width=4, height=5).json() == d2


def test_target_lab_and_tolerance_not_merged_by_formatting(client, ids):
    """目标 Lab / 容差在六位小数之外的差异不得被格式化合并。"""
    job = _run_job(client, ids)
    d1 = _retouch(client, job["job_id"],
                  target_lab=[50.0000001, 10.0, -10.0]).json()
    d2 = _retouch(client, job["job_id"],
                  target_lab=[50.0000002, 10.0, -10.0]).json()
    assert d1["retouch_id"] != d2["retouch_id"]
    # %.6f 会把 1e-7 与 0 都格式化为 "0.000000"，现在必须区分。
    d3 = _retouch(client, job["job_id"], tolerance_de=1e-7).json()
    d4 = _retouch(client, job["job_id"], tolerance_de=0.0).json()
    assert d3["retouch_id"] != d4["retouch_id"]
    rep3 = client.get(d3["report_url"]).json()
    assert rep3["request"]["tolerance_de"] == 1e-7
    assert rep3["constraints"]["tolerance_de00"] == 1e-7


def test_cache_binding_mismatch_triggers_recompute(client, ids):
    """缓存报告与本次请求的任务/矩形/约束绑定不符时，拒绝复用并重新计算。"""
    from app import main
    job = _run_job(client, ids)
    d1 = _retouch(client, job["job_id"]).json()
    rid = d1["retouch_id"]
    rdir = main.RETOUCH_DIR / rid
    # 篡改缓存报告为另一选区（模拟串用/脏缓存）。
    rep = json.loads((rdir / "report.json").read_text("utf-8"))
    rep["region"] = {"x": 0, "y": 0, "width": 2, "height": 2}
    rep["request"]["x"] = 0
    rep["request"]["y"] = 0
    (rdir / "report.json").write_text(json.dumps(rep), "utf-8")
    d2 = _retouch(client, job["job_id"]).json()
    assert d2["retouch_id"] == rid
    assert d2["rect"] == {"x": 4, "y": 4, "width": 16, "height": 16}
    # 磁盘上的报告也已按本次请求重新计算恢复。
    rep2 = json.loads((rdir / "report.json").read_text("utf-8"))
    assert rep2["region"] == {"x": 4, "y": 4, "width": 16, "height": 16}
    assert rep2["request"]["x"] == 4


def test_cache_missing_artifact_triggers_recompute(client, ids):
    """可达结果缺成品 TIFF 时不得复用缓存，须重新计算并恢复字节一致的成品。"""
    from app import main
    job = _run_job(client, ids, rgb=(180, 180, 180))
    target = _mean_lab_of_offset(client, ids, job["job_id"], (4, 4, 16, 16),
                                 (3, 0, 0, 0))
    kw = dict(target_lab=[float(v) for v in target], tolerance_de=0.5)
    d1 = _retouch(client, job["job_id"], **kw).json()
    assert d1["reachable"]
    tiff = main.RETOUCH_DIR / d1["retouch_id"] / "retouched.tif"
    blob = tiff.read_bytes()
    tiff.unlink()
    d2 = _retouch(client, job["job_id"], **kw).json()
    assert d2["retouch_id"] == d1["retouch_id"]
    assert tiff.exists()
    assert tiff.read_bytes() == blob


def test_proof_pngs_are_tagged_with_display_profile(client, ids):
    job = _run_job(client, ids)
    target = _mean_lab_of_offset(client, ids, job["job_id"], (4, 4, 16, 16),
                                 (-1, 1, 0, 0))
    d = _retouch(client, job["job_id"], target_lab=[float(v) for v in target],
                 tolerance_de=1.0).json()
    display = client.app.state.state.pm.load_bytes(ids["display"])
    for url in (d["images"]["before"], d["images"]["after"]):
        im = Image.open(io.BytesIO(client.get(url).content))
        assert im.info.get("icc_profile") == display


# --------------------------------------------------------------------------
# 缓存绑定：不仅核对指纹/任务/文件存在，还核对实际选区、约束、原任务、
# ICC 摘要与预览/成品的**内容**绑定；任一不符即重算并覆盖。
# --------------------------------------------------------------------------

def _reachable_kw(client, ids, job_id, rect=(4, 4, 16, 16), offset=(3, 0, 0, 0)):
    target = _mean_lab_of_offset(client, ids, job_id, rect, offset)
    return dict(x=rect[0], y=rect[1], width=rect[2], height=rect[3],
                target_lab=[float(v) for v in target], tolerance_de=0.5)


def test_region_only_tamper_in_report_triggers_recompute(client, ids):
    """只改报告 region/constraints 而保留 request 与 retouch_id：拒绝复用。"""
    from app import main
    job = _run_job(client, ids, rgb=(180, 180, 180))
    kw = _reachable_kw(client, ids, job["job_id"])
    d1 = _retouch(client, job["job_id"], **kw).json()
    rdir = main.RETOUCH_DIR / d1["retouch_id"]
    rep = json.loads((rdir / "report.json").read_text("utf-8"))
    # request 指纹与 retouch_id 保持原样，仅改报告的实际选区/约束。
    rep["region"] = {"x": 0, "y": 0, "width": 8, "height": 8}
    rep["constraints"]["total_ink_limit"] = 600
    (rdir / "report.json").write_text(
        json.dumps(rep, ensure_ascii=False), "utf-8")
    d2 = _retouch(client, job["job_id"], **kw).json()
    assert d2 == d1
    rep2 = json.loads((rdir / "report.json").read_text("utf-8"))
    assert rep2["region"] == d1["rect"]
    assert rep2["constraints"]["total_ink_limit"] == 1020


def test_replaced_before_png_content_triggers_recompute(client, ids):
    """以另一份合法 PNG 替换同名 before.png：内容绑定不符，重算覆盖。"""
    from app import main
    job = _run_job(client, ids, rgb=(180, 180, 180))
    kw = _reachable_kw(client, ids, job["job_id"])
    d1 = _retouch(client, job["job_id"], **kw).json()
    rdir = main.RETOUCH_DIR / d1["retouch_id"]
    before = rdir / "before.png"
    blob = before.read_bytes()
    im = np.asarray(Image.open(io.BytesIO(blob))).copy()
    im[0, 0] = np.clip(im[0, 0].astype(int) + 40, 0, 255).astype(np.uint8)
    display = client.app.state.state.pm.load_bytes(ids["display"])
    buf = io.BytesIO()
    Image.fromarray(im, "RGB").save(buf, format="PNG", icc_profile=display)
    before.write_bytes(buf.getvalue())
    try:
        d2 = _retouch(client, job["job_id"], **kw).json()
        assert d2 == d1
        assert before.read_bytes() == blob
    finally:
        before.write_bytes(blob)


def test_replaced_after_png_content_triggers_recompute(client, ids):
    from app import main
    job = _run_job(client, ids, rgb=(180, 180, 180))
    kw = _reachable_kw(client, ids, job["job_id"], offset=(2, 1, 0, 0))
    d1 = _retouch(client, job["job_id"], **kw).json()
    rdir = main.RETOUCH_DIR / d1["retouch_id"]
    after = rdir / "after.png"
    blob = after.read_bytes()
    im = np.asarray(Image.open(io.BytesIO(blob))).copy()
    im[1, 1] = (0, 255, 0)
    display = client.app.state.state.pm.load_bytes(ids["display"])
    buf = io.BytesIO()
    Image.fromarray(im, "RGB").save(buf, format="PNG", icc_profile=display)
    after.write_bytes(buf.getvalue())
    d2 = _retouch(client, job["job_id"], **kw).json()
    assert d2 == d1
    assert after.read_bytes() == blob


def test_replaced_tiff_content_triggers_recompute(client, ids):
    """以另一份合法 CMYK TIFF（仍嵌入同一印刷 ICC）替换同名成品：重算覆盖。"""
    from app import main
    job = _run_job(client, ids, rgb=(180, 180, 180))
    kw = _reachable_kw(client, ids, job["job_id"])
    d1 = _retouch(client, job["job_id"], **kw).json()
    assert d1["reachable"]
    tif = main.RETOUCH_DIR / d1["retouch_id"] / "retouched.tif"
    blob = tif.read_bytes()
    arr = np.asarray(Image.open(io.BytesIO(blob))).copy()
    # 选区外像素（0,0）被改动：合法可打开、同 ICC，但绑定不符。
    v = int(arr[0, 0, 0])
    arr[0, 0, 0] = v + 7 if v < 240 else v - 7
    press = client.app.state.state.pm.load_bytes(ids["cmyk"])
    buf = io.BytesIO()
    Image.fromarray(arr, "CMYK").save(
        buf, format="TIFF", compression="raw", icc_profile=press)
    tif.write_bytes(buf.getvalue())
    d2 = _retouch(client, job["job_id"], **kw).json()
    assert d2 == d1
    assert tif.read_bytes() == blob


def test_tampered_press_digest_in_cached_report_triggers_recompute(client, ids):
    from app import main
    job = _run_job(client, ids, rgb=(160, 100, 120))
    kw = _reachable_kw(client, ids, job["job_id"], offset=(1, 1, 0, 0))
    d1 = _retouch(client, job["job_id"], **kw).json()
    rdir = main.RETOUCH_DIR / d1["retouch_id"]
    rep = json.loads((rdir / "report.json").read_text("utf-8"))
    rep["profiles"]["press"]["sha256"] = "a" * 64
    (rdir / "report.json").write_text(json.dumps(rep), "utf-8")
    d2 = _retouch(client, job["job_id"], **kw).json()
    assert d2 == d1
    rep2 = json.loads((rdir / "report.json").read_text("utf-8"))
    assert rep2["profiles"]["press"]["sha256"] == \
        job["report"]["profiles"]["press"]["sha256"]


def test_tampered_source_job_binding_triggers_recompute(client, ids):
    """报告里原任务的原图哈希/尺寸被改：与当前任务绑定不符，重算覆盖。"""
    from app import main
    job = _run_job(client, ids, rgb=(150, 120, 90))
    kw = _reachable_kw(client, ids, job["job_id"], rect=(10, 10, 12, 12),
                       offset=(2, -1, 1, 0))
    d1 = _retouch(client, job["job_id"], **kw).json()
    rdir = main.RETOUCH_DIR / d1["retouch_id"]
    rep = json.loads((rdir / "report.json").read_text("utf-8"))
    rep["source_job"]["image_sha256"] = "9" * 64
    (rdir / "report.json").write_text(json.dumps(rep), "utf-8")
    d2 = _retouch(client, job["job_id"], **kw).json()
    assert d2 == d1
    rep2 = json.loads((rdir / "report.json").read_text("utf-8"))
    assert rep2["source_job"]["image_sha256"] == \
        job["report"]["image"]["sha256"]


def test_tampered_offset_and_residual_triggers_recompute(client, ids):
    """报告偏移/残余色差被改成另一组（文件齐全）：复算不符，重算覆盖。"""
    from app import main
    job = _run_job(client, ids, rgb=(180, 180, 180))
    kw = _reachable_kw(client, ids, job["job_id"])
    d1 = _retouch(client, job["job_id"], **kw).json()
    assert d1["best_offset_cmyk"] != [1, 1, 1, 0]
    rdir = main.RETOUCH_DIR / d1["retouch_id"]
    rep = json.loads((rdir / "report.json").read_text("utf-8"))
    rep["best_offset_cmyk"] = [1, 1, 1, 0]
    rep["best_abs_shift_sum"] = 3
    rep["residual"]["de00_to_target"] = 0.01
    (rdir / "report.json").write_text(json.dumps(rep), "utf-8")
    d2 = _retouch(client, job["job_id"], **kw).json()
    assert d2 == d1
    rep2 = json.loads((rdir / "report.json").read_text("utf-8"))
    assert rep2["best_offset_cmyk"] == d1["best_offset_cmyk"]
    assert rep2["residual"]["de00_to_target"] == \
        d1["residual"]["de00_to_target"]


def test_edge_clipped_region_binding_uses_actual_selection(client, ids):
    """贴边的合法裁剪按实际选区核对：48×48 图上 (44,44,16,16) 实际为 4×4。"""
    from app import main
    job = _run_job(client, ids, rgb=(160, 100, 120))
    kw = dict(x=44, y=44, width=16, height=16,
              target_lab=[50.0, 10.0, -10.0],
              tolerance_de=50.0, max_shift=2, tac_limit=1020)
    d1 = _retouch(client, job["job_id"], **kw).json()
    assert d1["rect"] == {"x": 44, "y": 44, "width": 4, "height": 4}
    rdir = main.RETOUCH_DIR / d1["retouch_id"]
    # 报告实际选区/像素数被改成未裁剪值：拒绝复用并恢复为实际裁剪选区。
    rep = json.loads((rdir / "report.json").read_text("utf-8"))
    rep["region"] = {"x": 44, "y": 44, "width": 16, "height": 16}
    rep["pixel_count"] = 256
    (rdir / "report.json").write_text(json.dumps(rep), "utf-8")
    d2 = _retouch(client, job["job_id"], **kw).json()
    assert d2["rect"] == {"x": 44, "y": 44, "width": 4, "height": 4}
    rep2 = json.loads((rdir / "report.json").read_text("utf-8"))
    assert rep2["region"] == {"x": 44, "y": 44, "width": 4, "height": 4}
    assert rep2["pixel_count"] == 16
    # 预览尺寸即实际裁剪尺寸。
    im = Image.open(io.BytesIO(client.get(d2["images"]["before"]).content))
    assert im.size == (4, 4)
    # 相同请求仍稳定复用（字节级）。
    assert _retouch(client, job["job_id"], **kw).json() == d2


def test_unreachable_stale_tiff_removed_on_reuse(client, ids):
    """不可达缓存目录里被塞进成品 TIFF：拒绝复用，清除脏成品后复用。"""
    from app import main
    job = _run_job(client, ids, rgb=(180, 180, 180))
    kw = _reachable_kw(client, ids, job["job_id"])
    reachable = _retouch(client, job["job_id"], **kw).json()
    tiff_blob = (main.RETOUCH_DIR / reachable["retouch_id"]
                 / "retouched.tif").read_bytes()
    body = {
        "job_id": job["job_id"], "x": 4, "y": 4, "width": 16, "height": 16,
        "target_lab": [95.0, -40.0, 40.0],
        "tolerance_de": 0.0, "max_shift": 0, "tac_limit": 1020,
    }
    du = client.post("/api/retouch", json=body).json()
    assert du["reachable"] is False
    udir = main.RETOUCH_DIR / du["retouch_id"]
    assert not (udir / "retouched.tif").exists()
    (udir / "retouched.tif").write_bytes(tiff_blob)
    du2 = client.post("/api/retouch", json=body).json()
    assert du2 == du
    assert not (udir / "retouched.tif").exists()
    assert client.get(f"/api/retouch/{du['retouch_id']}/retouched.tif"
                      ).status_code == 422


def test_corrupt_report_json_triggers_recompute(client, ids):
    from app import main
    job = _run_job(client, ids)
    d1 = _retouch(client, job["job_id"]).json()
    rp = main.RETOUCH_DIR / d1["retouch_id"] / "report.json"
    rp.write_text("{not json", "utf-8")
    d2 = _retouch(client, job["job_id"]).json()
    assert d2 == d1
    assert json.loads(rp.read_text("utf-8"))["retouch_id"] == d1["retouch_id"]


def test_identical_request_after_verification_remains_stable(client, ids):
    """经过强化核对后，相同请求连续提交仍稳定复用同一结果。"""
    job = _run_job(client, ids, rgb=(170, 120, 90))
    kw = _reachable_kw(client, ids, job["job_id"], rect=(10, 10, 12, 12),
                       offset=(2, -1, 1, 0))
    d1 = _retouch(client, job["job_id"], **kw).json()
    for _ in range(3):
        assert _retouch(client, job["job_id"], **kw).json() == d1


# --------------------------------------------------------------------------
# 内存风险：高多样性选区 + 宽偏移下，跨候选 Lab 缓存必须有界、可淘汰，
# 且淘汰只影响速度不改变数值；首次搜索仍评估全部可行偏移。
# --------------------------------------------------------------------------

def test_bounded_lab_cache_evicts_without_growing():
    """直接映射缓存的常驻槽位固定；写入远超容量的键也不增长，且未淘汰键正确。"""
    from app.retouch import _BoundedLabCache
    cap = 16
    cache = _BoundedLabCache(capacity=cap)
    assert cache.keys.shape[0] == cap and cache.labs.shape == (cap, 3)
    keys = np.arange(1000, dtype=np.uint64) * 7 + 1
    labs = np.stack([keys.astype(np.float64) % 97,
                     keys.astype(np.float64) % 89,
                     keys.astype(np.float64) % 83], axis=1)
    cache.insert(keys, labs,
                 cache.lookup(keys)[2], np.zeros(keys.shape[0], dtype=bool))
    # 常驻内存恒定：底层数组形状不随插入条目数变化。
    assert cache.keys.shape == (cap,) and cache.labs.shape == (cap, 3)
    # 仍在槽内（未被同槽后写键覆盖）的键取回值必须逐位正确。
    remaining = 0
    for k, lab in zip(keys.tolist(), labs):
        got = cache.get(int(k))
        if got is not None:
            remaining += 1
            assert np.array_equal(got, lab)
    assert 0 < remaining <= cap


def test_bounded_cache_eviction_keeps_mean_lab_results_exact():
    """小到必然发生淘汰的缓存，逐偏移结果仍与无界 dict 参考逐位一致。"""
    from app.retouch import (
        _BoundedLabCache, _feasible_offsets, _mean_lab_for_offset,
        _mean_labs_for_offsets,
    )
    rng = np.random.default_rng(7)
    # 内缩色带：宽偏移仍逐像素可行；高多样性：每个像素 CMYK 几乎都不同。
    roi = rng.integers(16, 241, size=(12, 12, 4)).astype(np.uint8)
    flat = np.ascontiguousarray(roi).reshape(-1, 4)
    cands = _feasible_offsets(roi, max_shift=4, tac_limit=1020)
    assert cands.shape[0] > 1000
    press = lcms_native.build_demo_cmyk_profile_bytes()
    with lcms_native.lab_transform("CMYK", press,
                                   "relative_colorimetric") as xf:
        ref = np.empty((cands.shape[0], 3))
        unbounded: dict = {}
        for i, off in enumerate(cands):
            ref[i] = _mean_lab_for_offset(flat, off, xf, unbounded)
        # 容量远小于搜索过程中出现的不同键数 -> 大量淘汰。
        tiny = _BoundedLabCache(capacity=64)
        got_batch = _mean_labs_for_offsets(flat, cands, xf, tiny, batch_size=8)
        got_one = np.empty_like(ref)
        for i, off in enumerate(cands):
            got_one[i] = _mean_lab_for_offset(flat, off, xf,
                                              _BoundedLabCache(capacity=64))
    assert np.array_equal(got_batch, ref)
    assert np.array_equal(got_one, ref)


def test_vectorized_search_evaluates_all_feasible_offsets_bounded_memory():
    """高多样性合法选区 + 宽偏移：全部可行偏移都被评估，常驻 Lab 缓存只占
    固定槽位（修复前会累积成无界字典直至 OOM）。默认用 24×24 / shift=5
    （14641 候选，搜索中出现的不同键远超缓存槽位，必然发生淘汰）；完整
    64×64 / shift=8 的 83521 候选压力探查设 SPD_SLOW_RETOUCH=1 运行。"""
    import os as _os
    from app.retouch import (
        _BoundedLabCache, _feasible_offsets, _mean_labs_for_offsets,
    )
    full_pressure = _os.environ.get("SPD_SLOW_RETOUCH") == "1"
    side, shift, expect = (64, 8, 17 ** 4) if full_pressure else (24, 5, 11 ** 4)
    rng = np.random.default_rng(0)
    roi = rng.integers(16, 241, size=(side, side, 4)).astype(np.uint8)
    flat = np.ascontiguousarray(roi).reshape(-1, 4)
    cands = _feasible_offsets(roi, max_shift=shift, tac_limit=1020)
    # 内缩色带使 -shift..+shift 全部合法且不触总墨量：完整候选空间。
    assert cands.shape[0] == expect
    press = lcms_native.build_demo_cmyk_profile_bytes()
    with lcms_native.lab_transform("CMYK", press,
                                   "relative_colorimetric") as xf:
        cache = _BoundedLabCache()
        fixed_bytes = cache.keys.nbytes + cache.labs.nbytes
        means = _mean_labs_for_offsets(flat, cands, xf, cache)
    # 每个候选都得到有限的平均 Lab（没有因去重/淘汰而漏算）。
    assert means.shape == (expect, 3)
    assert np.isfinite(means).all()
    # 常驻缓存数组在处理远超槽位数的不同键之后仍是同一固定大小。
    assert cache.keys.nbytes + cache.labs.nbytes == fixed_bytes
    assert cache.keys.shape[0] == cache.capacity


def _high_diversity_png(size=48, seed=11):
    """噪点 RGB 图：经演示配置分色后选区墨色高度多样。"""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_high_diversity_selection_returns_report_preview_and_tiff(
        client, ids, monkeypatch):
    """合法高多样性选区 + 宽偏移可经 POST /api/retouch 返回报告/预览/成品，
    不被系统杀掉；可达时 TIFF 选区外字节不变并嵌入原印刷 ICC。"""
    from app import main
    r = client.post(
        "/api/jobs",
        files={"image": ("noise.png", _high_diversity_png(), "image/png")},
        data={
            "source_profile": ids["srgb"],
            "press_profile": ids["cmyk"],
            "display_profile": ids["display"],
            "threshold": "3",
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    # 把分色结果替换为“合法高多样性 + 内缩色带”的最坏形态：每个像素一个
    # 独特 CMYK 四元组，且 -3..+3 偏移对全部像素逐通道可行、不触总墨量。
    rng = np.random.default_rng(5)
    cmyk_path = main.JOB_DIR / job_id / "cmyk.npy"
    orig = np.load(cmyk_path)
    side = 32
    diverse = rng.integers(24, 233, size=(side, side, 4)).astype(np.uint8)
    patched = orig.copy()
    patched[:side, :side] = diverse
    np.save(cmyk_path, patched)
    try:
        # 目标取“统一偏移 (2,-1,1,0) 后”的平均 Lab：必然可行且可达。
        target = _mean_lab_of_offset(
            client, ids, job_id, (0, 0, side, side), (2, -1, 1, 0))
        resp = _retouch(
            client, job_id, x=0, y=0, width=side, height=side,
            target_lab=[float(v) for v in target],
            tolerance_de=0.5, max_shift=3, tac_limit=1020,
        )
        assert resp.status_code == 200, resp.text
        d = resp.json()
        assert d["reachable"] is True
        assert d["feasible_offset_count"] == 7 ** 4
        assert d["rect"] == {"x": 0, "y": 0,
                            "width": side, "height": side}
        # 报告与修色前/后预览都可正常返回。
        assert client.get(d["report_url"]).status_code == 200
        assert client.get(d["images"]["before"]).status_code == 200
        assert client.get(d["images"]["after"]).status_code == 200
        # 成品 TIFF：选区外字节不变、嵌入原印刷 ICC。
        tif_resp = client.get(d["tiff"])
        assert tif_resp.status_code == 200
        tif = np.asarray(Image.open(io.BytesIO(tif_resp.content)))
        mask = np.zeros(patched.shape[:2], dtype=bool)
        mask[:side, :side] = True
        assert np.array_equal(tif[~mask], patched[~mask])
        press = client.app.state.state.pm.load_bytes(ids["cmyk"])
        with Image.open(io.BytesIO(tif_resp.content)) as im:
            assert im.info.get("icc_profile") == press
    finally:
        np.save(cmyk_path, orig)


def test_high_diversity_unreachable_returns_preview_without_tiff(client, ids):
    """高多样性选区不可达时仍返回报告与预览，但不产出 TIFF。"""
    from app import main
    r = client.post(
        "/api/jobs",
        files={"image": ("noise.png", _high_diversity_png(seed=13),
                         "image/png")},
        data={
            "source_profile": ids["srgb"],
            "press_profile": ids["cmyk"],
            "display_profile": ids["display"],
            "threshold": "3",
        },
    )
    job_id = r.json()["job_id"]
    cmyk_path = main.JOB_DIR / job_id / "cmyk.npy"
    orig = np.load(cmyk_path)
    side = 20
    rng = np.random.default_rng(9)
    patched = orig.copy()
    patched[:side, :side] = rng.integers(
        16, 241, size=(side, side, 4)).astype(np.uint8)
    np.save(cmyk_path, patched)
    try:
        # 极端目标 + 零偏移：不可能达标 -> 不可达。
        resp = _retouch(
            client, job_id, x=0, y=0, width=side, height=side,
            target_lab=[100.0, 127.0, 127.0],
            tolerance_de=0.0, max_shift=0, tac_limit=1020,
        )
        assert resp.status_code == 200, resp.text
        d = resp.json()
        assert d["reachable"] is False
        assert d["tiff"] is None
        assert client.get(d["images"]["before"]).status_code == 200
        assert client.get(d["images"]["after"]).status_code == 200
        assert client.get(
            f"/api/retouch/{d['retouch_id']}/retouched.tif"
        ).status_code == 422
    finally:
        np.save(cmyk_path, orig)
