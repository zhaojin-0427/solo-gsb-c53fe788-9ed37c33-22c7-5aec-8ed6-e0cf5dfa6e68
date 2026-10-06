"""FastAPI application: 印前色彩软打样台."""
from __future__ import annotations

import json
import hashlib
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import database as db
from . import pipeline as pl
from . import regions as regions_mod
from . import releases
from .release_api import router as release_router
from .config import (
    ACCEPTANCE_DIR,
    DB_PATH,
    JOB_DIR,
    MAX_DIMENSION,
    MAX_UPLOAD_MB,
    RETOUCH_DIR,
    init_state,
)
from .profiles import ProfileError
from . import acceptance as ac
from . import retouch as rt


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = init_state()
    releases.ensure_dirs()
    conn = db.connect(DB_PATH)
    db.init_db(conn)
    for rec in state.pm.list_profiles():
        db.upsert_profile(conn, rec)
    app.state.db = conn
    app.state.state = state
    yield
    conn.close()


app = FastAPI(title="印前色彩软打样台", version="1.3.0", lifespan=lifespan)
app.include_router(release_router)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def job_dir(job_id: str) -> Path:
    d = JOB_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_job_arrays(job_id: str) -> dict:
    base = JOB_DIR / job_id
    out = {}
    for name in ("lab_src", "lab_press", "cmyk", "de", "heatmap", "mask3"):
        out[name] = np.load(base / f"{name}.npy", mmap_mode="r")
    meta = json.loads((base / "meta.json").read_text("utf-8"))
    out["meta"] = meta
    return out


# ---------------------------------------------------------------------------
# API: health / profiles
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/profiles")
def list_profiles():
    state = app.state.state
    records = state.pm.list_profiles()
    for rec in records:
        db.upsert_profile(app.state.db, rec)
    groups = {"source": [], "press": [], "display": []}
    for rec in records:
        if rec.role in groups:
            d = rec.to_dict()
            groups[rec.role].append(d)
    for items in groups.values():
        items.sort(key=lambda d: (not d["builtin"], d["name"]))
    return groups


@app.post("/api/profiles/{role}")
async def upload_profile(role: str, file: UploadFile = File(...)):
    if role not in ("source", "press", "display"):
        raise HTTPException(400, "配置角色必须是 source / press / display")
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(400, "ICC 配置文件过大（上限 10MB）")
    state = app.state.state
    try:
        rec = state.pm.save_upload(role, file.filename or "uploaded.icc", data)
    except ProfileError as exc:
        raise HTTPException(422, str(exc))
    db.upsert_profile(app.state.db, rec)
    return rec.to_dict()


# ---------------------------------------------------------------------------
# API: run pipeline
# ---------------------------------------------------------------------------

@app.post("/api/jobs")
async def create_job(
    image: UploadFile = File(...),
    source_profile: str = Form(...),
    press_profile: str = Form(...),
    display_profile: str = Form(...),
    source_intent: str = Form("relative_colorimetric"),
    proof_intent: str = Form("relative_colorimetric"),
    threshold: float = Form(3.0),
    use_embedded: bool = Form(False),
):
    data = await image.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"图像超过上限 {MAX_UPLOAD_MB}MB")

    state = app.state.state
    inputs = pl.PipelineInputs(
        image_data=data,
        source_profile_id=source_profile,
        press_profile_id=press_profile,
        display_profile_id=display_profile,
        source_intent=source_intent,
        proof_intent=proof_intent,
        threshold=threshold,
        use_embedded=use_embedded,
        max_dimension=MAX_DIMENSION,
    )
    try:
        result = pl.run_pipeline(inputs, state.pm)
    except (pl.PipelineError, ProfileError) as exc:
        raise HTTPException(422, str(exc))

    job_id = pl.compute_job_id(result)
    base = job_dir(job_id)

    # Persist pixel arrays for region inspection at full double precision so
    # inspection and report statistics come from exactly the same Lab/ΔE
    # values that decided mask membership.
    np.save(base / "lab_src.npy", result.lab_src.astype(np.float64))
    np.save(base / "lab_press.npy", result.lab_press.astype(np.float64))
    np.save(base / "cmyk.npy", result.cmyk.astype(np.uint8))
    np.save(base / "de.npy", result.de.astype(np.float64))
    np.save(base / "heatmap.npy", result.heatmap.astype(np.uint8))
    np.save(base / "mask3.npy", result.mask.astype(np.uint8))
    (base / "original.bin").write_bytes(data)

    display_icc_bytes = state.pm.load_bytes(display_profile)
    pngs = pl.export_pngs(result, state.srgb_bytes, display_icc_bytes)
    for name, blob in pngs.items():
        (base / name).write_bytes(blob)

    report = pl.build_report(result, job_id)
    (base / "report.json").write_text(
        json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2), "utf-8"
    )

    meta = {
        "job_id": job_id,
        "width": result.img.size[0],
        "height": result.img.size[1],
        "mode": result.mode,
        "image_sha256": result.image_sha,
        "embedded_sha256": result.embedded_sha,
        "source_profile": source_profile,
        "press_profile": press_profile,
        "display_profile": display_profile,
        "source_intent": source_intent,
        "proof_intent": proof_intent,
        "threshold": threshold,
        "use_embedded": use_embedded,
        # 全 64 位配置摘要随任务固化：局部修色必须用同一 ICC 字节，
        # 摘要不符即拒绝。
        "source_profile_sha256": result.profile_summaries["source"]["sha256"],
        "press_profile_sha256": result.profile_summaries["press"]["sha256"],
        "display_profile_sha256": result.profile_summaries["display"]["sha256"],
    }
    (base / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), "utf-8")

    db.insert_job(app.state.db, {
        "job_id": job_id,
        "image_sha256": result.image_sha,
        "width": result.img.size[0],
        "height": result.img.size[1],
        "image_mode": result.mode,
        "source_profile": source_profile if not use_embedded else "<embedded>",
        "press_profile": press_profile,
        "display_profile": display_profile,
        "source_intent": source_intent,
        "proof_intent": proof_intent,
        "threshold": threshold,
        "use_embedded": use_embedded,
        "embedded_sha256": result.embedded_sha,
        "parameters": report["parameters"],
        "profile_digests": {k: v["sha256"] for k, v in result.profile_summaries.items()},
        "stats": result.stats,
    })

    return {
        "job_id": job_id,
        "width": result.img.size[0],
        "height": result.img.size[1],
        "stats": result.stats,
        "report": report,
        "images": {
            "preview": f"/api/jobs/{job_id}/images/preview.png",
            "softproof": f"/api/jobs/{job_id}/images/softproof.png",
            "heatmap": f"/api/jobs/{job_id}/images/heatmap.png",
            "overlay": f"/api/jobs/{job_id}/images/overlay.png",
            "mask": f"/api/jobs/{job_id}/images/mask.png",
        },
    }


# ---------------------------------------------------------------------------
# API: job artifacts
# ---------------------------------------------------------------------------

def _require_job(job_id: str) -> Path:
    base = JOB_DIR / job_id
    if not (base / "meta.json").exists():
        raise HTTPException(404, "任务不存在或已被删除")
    return base


@app.get("/api/jobs")
def list_jobs():
    rows = db.recent_jobs(app.state.db)
    return [dict(r) for r in rows]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    _require_job(job_id)
    return JSONResponse(json.loads((JOB_DIR / job_id / "report.json").read_text("utf-8")))


@app.get("/api/jobs/{job_id}/images/{name}")
def get_job_image(job_id: str, name: str):
    base = _require_job(job_id)
    allowed = {"preview.png", "softproof.png", "heatmap.png", "overlay.png", "mask.png"}
    if name not in allowed:
        raise HTTPException(404, "图像不存在")
    return FileResponse(base / name, media_type="image/png")


@app.get("/api/jobs/{job_id}/report")
def get_job_report(job_id: str):
    base = _require_job(job_id)
    return FileResponse(
        base / "report.json", media_type="application/json", filename="report.json"
    )


@app.get("/api/jobs/{job_id}/export.zip")
def get_job_zip(job_id: str):
    base = _require_job(job_id)
    cache = base / "export.zip"
    if not cache.exists():
        pngs = {}
        for name in ("preview.png", "softproof.png", "heatmap.png", "overlay.png", "mask.png"):
            pngs[name] = (base / name).read_bytes()
        report = json.loads((base / "report.json").read_text("utf-8"))
        cache.write_bytes(pl.export_zip(pngs, report))
    return FileResponse(cache, media_type="application/zip", filename=f"{job_id[:12]}-softproof.zip")


@app.post("/api/jobs/{job_id}/region")
async def inspect_region(job_id: str, payload: dict):
    _require_job(job_id)
    try:
        arrays = load_job_arrays(job_id)
    except FileNotFoundError:
        raise HTTPException(404, "任务像素数据不存在")

    class R:  # light adapter
        pass

    r = R()
    r.img = type("I", (), {"size": (arrays["meta"]["width"], arrays["meta"]["height"])})()
    r.lab_src = np.asarray(arrays["lab_src"])
    r.lab_press = np.asarray(arrays["lab_press"])
    r.cmyk = np.asarray(arrays["cmyk"])
    r.de = np.asarray(arrays["de"])
    r.inputs = type("P", (), {"threshold": arrays["meta"]["threshold"]})()

    try:
        return regions_mod.inspect_region(
            r,
            int(payload.get("x", 0)),
            int(payload.get("y", 0)),
            int(payload.get("width", 10)),
            int(payload.get("height", 10)),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))


# ---------------------------------------------------------------------------
# API: 局部 CMYK 修色（以已保存任务分色结果为起点）
# ---------------------------------------------------------------------------

def _retouch_dir(retouch_id: str) -> Path:
    d = RETOUCH_DIR / retouch_id
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.post("/api/retouch")
def create_retouch(payload: dict | None = None):
    if not isinstance(payload, dict):
        raise HTTPException(422, "请求体必须是 JSON 对象")
    try:
        req = rt.validate_request(payload)
    except rt.RetouchError as exc:
        raise HTTPException(422, str(exc))

    job_base = JOB_DIR / req.job_id
    if not (job_base / "meta.json").exists():
        raise HTTPException(404, "原任务不存在或已被删除")

    try:
        ctx = rt.load_job_context(job_base)
    except rt.RetouchError as exc:
        raise HTTPException(422, str(exc))
    meta = ctx["meta"]

    state = app.state.state
    # 原任务印刷 ICC：必须存在且 SHA-256 与任务记录严格一致。
    row = db.get_job(app.state.db, req.job_id)
    recorded_press_digest = None
    if row is not None and row["profile_digests_json"]:
        try:
            recorded_press_digest = json.loads(row["profile_digests_json"]).get("press")
        except (json.JSONDecodeError, TypeError, AttributeError):
            recorded_press_digest = None
    recorded_press_digest = recorded_press_digest or meta.get("press_profile_sha256")

    press_profile_id = meta["press_profile"]
    try:
        press_bytes = state.pm.load_bytes(press_profile_id)
        rt.verify_press_profile(press_bytes, recorded_press_digest)
        # 摘要通过后补名称等元数据，报告仍以哈希为绑定凭据。
        press_summary_full = pl.summary_from_bytes(press_bytes, press_profile_id)
    except ProfileError as exc:
        raise HTTPException(422, f"原任务印刷 ICC 缺失，无法修色: {exc}")
    except rt.RetouchError as exc:
        raise HTTPException(422, str(exc))

    display_profile_id = meta["display_profile"]
    recorded_display_digest = meta.get("display_profile_sha256")
    try:
        display_bytes = state.pm.load_bytes(display_profile_id)
        display_summary_full = pl.summary_from_bytes(display_bytes, display_profile_id)
    except ProfileError as exc:
        raise HTTPException(422, f"原任务显示 ICC 缺失，无法生成软打样: {exc}")
    if recorded_display_digest and display_summary_full["sha256"] != recorded_display_digest:
        raise HTTPException(
            422,
            "原任务显示 ICC 摘要不符，无法按原显示配置呈现修色软打样，拒绝修色",
        )

    retouch_id = rt.compute_retouch_id(meta, req)
    fingerprint = rt.request_fingerprint(meta, req)
    base = _retouch_dir(retouch_id)
    report_path = base / "report.json"
    report = None
    if report_path.exists():
        try:
            cached = json.loads(report_path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = None
        # 命中缓存必须逐项核对报告的实际选区（贴边裁剪按实际交点）、约束、
        # 原任务与印刷/显示 ICC 摘要，并按当前分色与 ICC 重渲染/重导后比对
        # 预览 PNG 与成品 TIFF 的内容绑定；任一不符（含文件被另一份有效
        # 同名文件替换）即拒绝复用，落入下方按本次请求重新计算并覆盖。
        if cached is not None and rt.verify_cache(
            cached, retouch_id, fingerprint, base,
            ctx, req, press_bytes, display_bytes,
        ):
            report = cached
    if report is None:
        try:
            result = rt.run_retouch(req, ctx, press_bytes)
        except rt.RetouchError as exc:
            raise HTTPException(422, str(exc))
        except RuntimeError as exc:
            raise HTTPException(422, f"LittleCMS 色彩转换失败: {exc}")

        try:
            pngs = rt.render_proof_crops(
                result, ctx["cmyk"], press_bytes, display_bytes,
                meta.get("proof_intent", "relative_colorimetric"),
            )
        except RuntimeError as exc:
            raise HTTPException(422, f"软打样渲染失败: {exc}")
        (base / "before.png").write_bytes(pngs["before.png"])
        (base / "after.png").write_bytes(pngs["after.png"])

        report = rt.build_report(
            retouch_id, meta, press_summary_full, display_summary_full, req, result
        )
        if result["reachable"]:
            tiff = rt.export_retouched_tiff(result["_cmyk_adjusted"], press_bytes)
            (base / "retouched.tif").write_bytes(tiff)
        else:
            # 不可达：显式不生成成品（若历史脏文件存在则移除）。
            stale = base / "retouched.tif"
            if stale.exists():
                stale.unlink()

        report_path.write_text(
            json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2),
            "utf-8",
        )

    return {
        "retouch_id": retouch_id,
        "reachable": report["reachable"],
        "rect": report["region"],
        "best_offset_cmyk": report["best_offset_cmyk"],
        "best_abs_shift_sum": report["best_abs_shift_sum"],
        "feasible_offset_count": report["search"]["feasible_offset_count"],
        "residual": report["residual"],
        "constraints": report["constraints"],
        "images": {
            "before": f"/api/retouch/{retouch_id}/before.png",
            "after": f"/api/retouch/{retouch_id}/after.png",
        },
        "tiff": (
            f"/api/retouch/{retouch_id}/retouched.tif"
            if report["reachable"] else None
        ),
        "report_url": f"/api/retouch/{retouch_id}/report",
        "report": report,
    }


def _require_retouch(retouch_id: str) -> Path:
    base = RETOUCH_DIR / retouch_id
    if not (base / "report.json").exists():
        raise HTTPException(404, "修色任务不存在")
    return base


@app.get("/api/retouch/{retouch_id}/report")
def get_retouch_report(retouch_id: str):
    base = _require_retouch(retouch_id)
    return FileResponse(
        base / "report.json", media_type="application/json", filename="retouch-report.json"
    )


@app.get("/api/retouch/{retouch_id}/{name}")
def get_retouch_artifact(retouch_id: str, name: str):
    base = _require_retouch(retouch_id)
    if name in ("before.png", "after.png"):
        return FileResponse(base / name, media_type="image/png")
    if name == "retouched.tif":
        report = json.loads((base / "report.json").read_text("utf-8"))
        if not report.get("reachable") or not (base / "retouched.tif").exists():
            # 不可达不产出成品。
            raise HTTPException(422, "无可行解或未达容差：标记为不可达，未生成成品 TIFF")
        return FileResponse(
            base / "retouched.tif", media_type="image/tiff",
            filename=f"retouched-{retouch_id[:12]}.tif",
        )
    raise HTTPException(404, "产物不存在")


# ---------------------------------------------------------------------------
# API: 印后测色验收（post-press color acceptance）
# ---------------------------------------------------------------------------

def _acceptance_dir(acceptance_id: str) -> Path:
    d = ACCEPTANCE_DIR / acceptance_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _is_sha256_id(value: str) -> bool:
    return bool(value) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value
    )


@app.get("/api/jobs/{job_id}/retouches")
def list_job_retouches(job_id: str):
    """列出该已保存任务下可用于印后验收的**达标**修色成品。"""
    _require_job(job_id)
    return ac.list_retouches_for_job(RETOUCH_DIR, job_id)


@app.post("/api/acceptance")
async def create_acceptance(
    job_id: str = Form(...),
    limit_de: float = Form(...),
    file: UploadFile = File(...),
    retouch_id: str | None = Form(None),
):
    if not isinstance(limit_de, (int, float)) or isinstance(limit_de, bool):
        raise HTTPException(422, "ΔE00 上限必须是非负数值")
    limit_de = float(limit_de)
    if not np.isfinite(limit_de) or limit_de < 0.0:
        raise HTTPException(422, "ΔE00 上限必须是非负有限数值")

    csv_bytes = await file.read()

    job_base = JOB_DIR / job_id
    if not (job_base / "meta.json").exists():
        raise HTTPException(404, "任务不存在或已被删除")

    rid = (retouch_id or "").strip() or None
    if rid == "":
        rid = None
    if rid is not None and not _is_sha256_id(rid):
        raise HTTPException(422, "retouch_id 必须是 64 位十六进制内容哈希")
    retouch_base = RETOUCH_DIR / rid if rid is not None else None
    if rid is not None and not (retouch_base / "report.json").exists():
        raise HTTPException(404, "所选修色任务不存在或已被删除")

    try:
        ctx = ac.load_job(job_base)
    except ac.AcceptanceNotFound as exc:
        raise HTTPException(404, str(exc))
    except ac.AcceptanceError as exc:
        raise HTTPException(422, str(exc))
    meta = ctx["meta"]

    # 严格按任务尺寸校验 CSV（越界一律拒绝）。
    try:
        points = ac.parse_measurements(
            csv_bytes, int(meta["width"]), int(meta["height"])
        )
    except ac.AcceptanceError as exc:
        raise HTTPException(422, str(exc))

    # 原任务印刷 ICC：必须存在且摘要与任务固化记录严格一致。
    state = app.state.state
    row = db.get_job(app.state.db, job_id)
    digest = ac.recorded_press_digest(row, meta)
    try:
        press_bytes = state.pm.load_bytes(meta["press_profile"])
        ac.verify_press(press_bytes, digest)
        press_summary_full = pl.summary_from_bytes(
            press_bytes, meta["press_profile"]
        )
    except ProfileError as exc:
        raise HTTPException(422, f"原任务印刷 ICC 缺失，无法验收: {exc}")
    except ac.AcceptanceError as exc:
        raise HTTPException(422, str(exc))

    # 可选修色成品：独立内容绑定核验；不可达/绑定不符一律拒绝，不回退。
    retouch_report = None
    verified_tiff = None
    if rid is not None:
        try:
            retouch_report = ac.verify_retouch_binding(
                rid, retouch_base, ctx, ctx["cmyk"], press_bytes
            )
            with Image.open(retouch_base / "retouched.tif") as im:
                im.load()
                verified_tiff = np.asarray(im)
        except ac.AcceptanceNotFound as exc:
            raise HTTPException(404, str(exc))
        except ac.AcceptanceError as exc:
            raise HTTPException(422, str(exc))
        except Exception as exc:  # 成品无法解码等同绑定不符，拒绝验收。
            raise HTTPException(422, f"修色成品无法核验: {exc}")

    csv_sha = hashlib.sha256(csv_bytes).hexdigest()
    # 身份指纹包含原始 CSV 字节摘要：同一规范测点序列的不同原件（仅表头别名/
    # 空白/BOM 等不同）各自寻址，得到各自的 acceptance_id、报告与原样 CSV。
    fingerprint = ac.request_fingerprint(
        meta, points, limit_de, rid, digest, csv_sha
    )
    acceptance_id = ac.compute_acceptance_id(fingerprint)
    base = _acceptance_dir(acceptance_id)
    report_path = base / "report.json"
    csv_path = base / "measurements.csv"

    report = None
    if report_path.exists():
        try:
            cached = json.loads(report_path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = None
        if cached is not None and ac.verify_report_binding(
            cached, acceptance_id, fingerprint, ctx, points, limit_de,
            press_bytes, press_summary_full, rid, retouch_report,
            verified_tiff, csv_sha,
        ):
            # 报告内容与本次提交逐项一致（含原件摘要）。归档 measurements.csv
            # 若缺失或被另一份字节替换（与报告摘要不符），用本次提交的、摘要
            # 已核验相同的字节原样修复后再复用——不静默交付错配原件。
            if not ac.stored_csv_matches_report(cached, csv_path):
                csv_path.write_bytes(csv_bytes)
            report = cached
        elif cached is not None:
            # 内容寻址目录出现绑定不符的脏文件：拒绝验收而不是静默覆盖。
            raise HTTPException(
                422, "已存在的验收报告与当前任务/ICC/修色绑定核验不符，拒绝验收"
            )

    if report is None:
        evaluation = ac.evaluate(
            points, limit_de, ctx, press_bytes, press_summary_full,
            rid, retouch_report, verified_tiff,
        )
        report = ac.build_report(
            acceptance_id, fingerprint, ctx, points, limit_de,
            rid, retouch_report, evaluation, csv_sha,
        )
        csv_path.write_bytes(csv_bytes)
        report_path.write_text(
            json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2),
            "utf-8",
        )

    return {
        "acceptance_id": acceptance_id,
        "passed": report["passed"],
        "job_id": job_id,
        "retouch_id": rid,
        "point_count": report["point_count"],
        "over_limit_count": report["over_limit_count"],
        "over_limit_points": report["over_limit_points"],
        "limit_de00": report["rule"]["limit_de00"],
        "stats": report["stats"],
        "prediction_method": report["prediction_method"],
        "points": report["points"],
        "report_url": f"/api/acceptance/{acceptance_id}/report",
        "softproof_url": f"/api/jobs/{job_id}/images/softproof.png",
        "report": report,
    }


def _require_acceptance(acceptance_id: str) -> Path:
    if not _is_sha256_id(acceptance_id):
        raise HTTPException(404, "验收报告不存在")
    base = ACCEPTANCE_DIR / acceptance_id
    if not (base / "report.json").exists():
        raise HTTPException(404, "验收报告不存在")
    return base


@app.get("/api/acceptance/{acceptance_id}")
def get_acceptance(acceptance_id: str):
    base = _require_acceptance(acceptance_id)
    return JSONResponse(json.loads((base / "report.json").read_text("utf-8")))


@app.get("/api/acceptance/{acceptance_id}/report")
def get_acceptance_report(acceptance_id: str):
    base = _require_acceptance(acceptance_id)
    return FileResponse(
        base / "report.json", media_type="application/json",
        filename=f"acceptance-{acceptance_id[:12]}.json",
    )


@app.get("/api/acceptance/{acceptance_id}/measurements.csv")
def get_acceptance_csv(acceptance_id: str):
    base = _require_acceptance(acceptance_id)
    csv_path = base / "measurements.csv"
    # 摘要核验与响应交付必须使用同一份字节：先读出归档原件（缺失/不可读
    # 按 404），核验通过后直接交付这份内存字节——不再按路径二次打开，也不
    # 依赖已打开的句柄（原位改写对二者都不可防御）。若核验后归档被替换或
    # 原位改写，交出的仍是与报告摘要一致的那份字节。
    blob = ac.read_stored_csv(csv_path)
    if blob is None:
        raise HTTPException(404, "验收测点 CSV 不存在")
    try:
        report = json.loads((base / "report.json").read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        raise HTTPException(422, "验收报告不可读，无法核验测点原件摘要")
    # 交出原件前核对其 SHA-256 与报告固化摘要：归档 CSV 被另一份字节
    # 替换时绝不下载与报告摘要不符的文件。
    if not ac.csv_bytes_match_report(report, blob):
        raise HTTPException(
            422, "归档测点 CSV 与验收报告记录的 SHA-256 摘要不符，拒绝下载"
        )
    return Response(
        content=blob,
        media_type="text/csv",
        headers={
            "Content-Disposition":
                f'attachment; filename="measurements-{acceptance_id[:12]}.csv"'
        },
    )


# Static frontend (mounted last so /api routes win).
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
