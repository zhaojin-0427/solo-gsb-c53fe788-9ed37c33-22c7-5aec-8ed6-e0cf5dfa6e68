"""印后测色验收（post-press color acceptance）。

选择一个**已保存软打样任务**，或该任务下一个**达标（reachable）的局部修色
成品**，上传非空 CSV 测点（零基整数 ``x/y`` + 有限 D50 Lab：
``L ∈ [0,100]``、``a/b ∈ [-128,127]``），设置非负 ΔE00 上限后逐点验收：

* **不使用修色**：逐点预测 Lab 直接取原任务 ``lab_press.npy``——它是分色
  CMYK 经原印刷 ICC 回算的逐像素预测值；
* **绑定达标修色成品**：先对成品做**独立的内容绑定核验**（报告绑定本任务、
  可达、印刷 ICC 摘要一致、报告 ``request`` 指纹与 retouch_id 严格相符；
  ``retouched.tif`` 解码后选区外与原分色逐字节相同、选区内恰为报告所记录的
  统一整数偏移、嵌入的印刷 ICC 字节正确），核验通过后：

  - 修色**选区内**测点：从**已核验成品**取该点 CMYK，按**原印刷 ICC 与原
    任务意图**回算为 Lab（策略 ``verified_product_backcalc``——成品选区经
    核验恰为“原分色 + 统一偏移”，因此与“按原分色与偏移重建”在字节与数值
    上等价，这里显式选择从已核验成品回算）；
  - 修色**选区外**测点：保持原任务预测值（``lab_press.npy``）。

**修色选区的整数语义（像素校验与逐点预测共用同一已验证选区）**：

修色请求指纹（报告 ``request``）记录的是用户请求矩形；报告 ``region`` 记录
的是该矩形与原任务幅面的**实际交点**——请求矩形贴住右/下边缘时按实际交点
裁剪（如 48×48 图上请求 ``(44,44,16,16)`` 的实际选区为 ``(44,44,4,4)``）。
验收绑定时：

1. 先严格核验报告 ``request`` 就是 retouch_id 的原像（规范 JSON 之 SHA-256
   必须等于所绑 retouch_id），且请求中的 ``x/y/width/height/max_shift/
   tac_limit`` 均为**非布尔整数**、落在修色接口的同一合法范围内，
   ``job_id``/渲染意图与本任务一致；
2. 用**该请求矩形与当前原任务幅面重新求交**得到唯一的“已验证选区”，报告
   ``region`` 必须与它**逐字段精确相等**（不许多余键、不许浮点/布尔、不许
   贴边裁剪以外的任何偏差——归档报告把 ``x=4,width=1`` 改成 ``x=4.5,
   width=1.5`` 会在此直接 422，而不是被 ``int()`` 截断默许）；
3. ``best_offset_cmyk`` 必须是 4 个**非布尔整数**且对已验证选区逐像素满足
   字节范围 / 单通道上限 / 总墨量上限（与修色搜索同一可行口径）；
4. **TIFF 逐像素核验与逐点“选区内外”判定使用同一个已验证选区**，绝不允许
   一边按截断整数核像素、一边按报告原值划内外。

随后把预测 Lab 与 CSV 实测 Lab 逐点计算 CIEDE2000；同坐标复测逐条保留、
不去重。**任一点 ΔE00 严格大于上限即整张不合格**。

**身份与原件**：验收身份（acceptance_id）的指纹不仅包含规范后的测点序列，
还包含**原始 CSV 字节的 SHA-256**。两份 CSV 即使解析出完全相同的测点（仅
表头别名写法、空白/换行/BOM 等不同），也是**各自独立、分别寻址**的验收：
得到各自的 acceptance_id、各自绑定的报告与原样保存的 measurements.csv，
绝不共享目录，因此不存在“同身份下原件错配”。逐点结论只取决于规范测点，
故两份提交的逐点结果相同；完全相同的 CSV 字节重复提交稳定命中同一目录
复用。下载 measurements.csv 时先读出归档字节、核对其 SHA-256 与报告固化
摘要一致，随后**直接交付这份已核验字节**（不再按路径二次打开，也不依赖
已打开的句柄——原位改写对二者都不可防御）：归档在核验后被替换或原位改写，
也不会交出与报告不符的内容；摘要不符一律拒绝下载。命中缓存时若发现归档
原件缺失或摘要不符，则用本次提交的同摘要字节原样修复后复用。

下列情况一律拒绝验收（422/404），绝不用原任务值代替修色测点：修色不可达、
修色报告或成品与任务/ICC 绑定不符、报告指纹与 retouch_id 不符、报告选区/
偏移不是与请求相符的严格整数记录（含浮点/布尔坐标或偏移、与贴边裁剪结果
不符的选区）、原印刷 ICC 缺失或摘要不符、测点越界、CSV 非法。
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import colorscience, lcms_native, retouch as rt

PRESS_INTENT_FIELD = "source_intent"


class AcceptanceError(ValueError):
    """请求参数 / 绑定不合法（HTTP 422）。"""


class AcceptanceNotFound(LookupError):
    """引用的任务 / 修色产物不存在（HTTP 404）。"""


@dataclass(frozen=True)
class Measurement:
    x: int
    y: int
    L: float
    a: float
    b: float


@dataclass(frozen=True)
class VerifiedRegion:
    """经报告指纹 + 幅面求交双重验证的唯一修色选区。

    TIFF 像素核验与逐点“选区内外”判定必须共用这同一个对象，杜绝“截断
    整数核像素、原始值划内外”的解释分歧。
    """
    x: int
    y: int
    width: int
    height: int
    offset: tuple[int, int, int, int]

    @property
    def rect(self) -> dict:
        return {"x": self.x, "y": self.y,
                "width": self.width, "height": self.height}


# ---------------------------------------------------------------------------
# CSV 解析（严格：零基整数 x/y、有限 D50 Lab）
# ---------------------------------------------------------------------------

# 表头规范化：去空白/下划线/星号后小写。L* 的星号、L_a_b 之类写法都能识别。
_HEADER_ALIASES = {
    "x": "x",
    "y": "y",
    "l": "L",
    "a": "a",
    "b": "b",
}
_REQUIRED_COLUMNS = ("x", "y", "L", "a", "b")
# 单次验收允许的最大测点行数（防止超大上传做无界转换）。
MAX_POINTS = 200_000
# 上传 CSV 的字节上限。
MAX_CSV_BYTES = 20 * 1024 * 1024


def _norm_header(token: str) -> str:
    return token.strip().lower().replace("_", "").replace("*", "")


def parse_measurements(data: bytes, width: int, height: int) -> list[Measurement]:
    """严格解析测点 CSV；任何非法内容抛 AcceptanceError。

    要求：UTF-8（可带 BOM）文本；首行为表头，规范化后必须且只能包含
    x,y,L,a,b 五列（顺序任意、大小写/星号/空白宽容）；至少一行数据；
    x/y 为零基整数且落在 0..width-1 / 0..height-1；L/a/b 为有限数值且在
    D50 Lab 合法范围内。同坐标复测允许重复、逐条保留。
    """
    if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
        raise AcceptanceError("测点 CSV 为空：请上传至少含一个测点的非空 CSV 文件")
    if len(data) > MAX_CSV_BYTES:
        raise AcceptanceError(f"测点 CSV 超过 {MAX_CSV_BYTES // (1024 * 1024)}MB 上限")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AcceptanceError(f"测点 CSV 必须是 UTF-8 文本，解码失败: {exc}")

    try:
        reader = csv.reader(io.StringIO(text))
        all_rows = list(reader)
    except csv.Error as exc:
        raise AcceptanceError(f"测点 CSV 解析失败: {exc}")

    # 保留原始行号（1 起算），仅统计非空行；表头之外的非空行才是测点。
    numbered = [
        (lineno, row)
        for lineno, row in enumerate(all_rows, start=1)
        if any(cell.strip() != "" for cell in row)
    ]
    if not numbered:
        raise AcceptanceError("测点 CSV 为空：缺少表头与测点数据")

    header_lineno, header = numbered[0]
    if not header:
        raise AcceptanceError("测点 CSV 缺少表头行（需要 x,y,L,a,b 五列）")
    index: dict[str, int] = {}
    for i, raw in enumerate(header):
        key = _norm_header(raw)
        if key not in _HEADER_ALIASES:
            raise AcceptanceError(
                f"测点 CSV 含无法识别的表头列 {raw.strip()!r}：只允许 x,y,L,a,b 五列"
            )
        col = _HEADER_ALIASES[key]
        if col in index:
            raise AcceptanceError(f"测点 CSV 表头列 {col} 重复")
        index[col] = i
    missing = [c for c in _REQUIRED_COLUMNS if c not in index]
    if missing:
        raise AcceptanceError(f"测点 CSV 缺少必需列: {', '.join(missing)}")
    if len(header) != len(_REQUIRED_COLUMNS):
        raise AcceptanceError("测点 CSV 必须且只能包含 x,y,L,a,b 五列")

    points: list[Measurement] = []
    for lineno, row in numbered[1:]:
        if len(row) != len(_REQUIRED_COLUMNS):
            raise AcceptanceError(
                f"测点 CSV 第 {lineno} 行列数不为 5（需要恰好 x,y,L,a,b）"
            )
        # 坐标单元格保留原始内容交给严格整数解析（拒绝内嵌空白/+1/1.0）；
        # Lab 数值由 float() 宽容首尾空白，空值在各解析函数内拒绝。
        raw = [row[index[col]] for col in _REQUIRED_COLUMNS]
        px = _parse_coord(raw[0], "x", width, lineno)
        py = _parse_coord(raw[1], "y", height, lineno)
        L = _parse_lab(raw[2].strip(), "L", 0.0, 100.0, lineno)
        a = _parse_lab(raw[3].strip(), "a", -128.0, 127.0, lineno)
        b = _parse_lab(raw[4].strip(), "b", -128.0, 127.0, lineno)
        points.append(Measurement(px, py, L, a, b))
        if len(points) > MAX_POINTS:
            raise AcceptanceError(f"测点数超过上限 {MAX_POINTS} 行")

    if not points:
        raise AcceptanceError(
            f"测点 CSV 第 {header_lineno} 行后没有任何数据行（至少需要一个测点）"
        )
    return points


def _parse_coord(token: str, name: str, hi: int, lineno: int) -> int:
    if token == "":
        raise AcceptanceError(f"测点 CSV 第 {lineno} 行 {name} 为空")
    # 只接受十进制整数字面量：拒绝 1.0 / 1.5 / 0x1 / +1 / 内嵌空白等写法。
    s = token
    if not s or not s.isascii() or not s.isdigit():
        raise AcceptanceError(f"测点 CSV 第 {lineno} 行 {name}={token!r} 不是零基整数")
    v = int(s)
    if v >= hi:
        raise AcceptanceError(
            f"测点 CSV 第 {lineno} 行 {name}={v} 越界：该任务该维度仅 0..{hi - 1}"
        )
    return v


def _parse_lab(token: str, name: str, lo: float, hi: float, lineno: int) -> float:
    if token == "":
        raise AcceptanceError(f"测点 CSV 第 {lineno} 行 {name} 为空")
    try:
        v = float(token)
    except ValueError:
        raise AcceptanceError(f"测点 CSV 第 {lineno} 行 {name}={token!r} 不是数值")
    if not np.isfinite(v):
        raise AcceptanceError(f"测点 CSV 第 {lineno} 行 {name}={token} 不是有限数值")
    if not (lo <= v <= hi):
        raise AcceptanceError(f"测点 CSV 第 {lineno} 行 {name}={v} 超出允许范围 {lo}..{hi}")
    return float(v)


# ---------------------------------------------------------------------------
# 任务 / 修色成品加载与绑定核验
# ---------------------------------------------------------------------------

def recorded_press_digest(db_row, meta: dict) -> str:
    """取任务创建时固化的全 64 位印刷 ICC 摘要（数据库行优先，回退 meta）。"""
    digest = None
    if db_row is not None:
        raw = db_row["profile_digests_json"] if "profile_digests_json" in db_row.keys() else None
        if raw:
            try:
                digest = json.loads(raw).get("press")
            except (json.JSONDecodeError, TypeError, AttributeError):
                digest = None
    return digest or meta.get("press_profile_sha256")


def load_job(job_base: Path):
    """加载原任务上下文与逐像素预测值；缺失即拒绝。"""
    if not (job_base / "meta.json").exists():
        raise AcceptanceNotFound("任务不存在或已被删除")
    try:
        ctx = rt.load_job_context(job_base)
    except rt.RetouchError as exc:
        raise AcceptanceError(str(exc))
    try:
        lab_press = np.load(job_base / "lab_press.npy", mmap_mode="r")
    except FileNotFoundError:
        raise AcceptanceError("原任务 lab_press.npy 缺失，无法取得逐像素预测值")
    ctx["lab_press"] = lab_press
    return ctx


def verify_press(press_bytes: bytes, digest: str | None) -> None:
    """原印刷 ICC 必须存在、可打开且 SHA-256 与任务固化摘要严格一致。"""
    try:
        rt.verify_press_profile(press_bytes, digest)
    except rt.RetouchError as exc:
        raise AcceptanceError(str(exc))


def _strict_int(value, name: str) -> int:
    """严格整数：拒绝 bool（``True`` 会静默等于 1）与 float（含 1.0/4.5）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise AcceptanceError(f"修色报告的 {name} 必须是整数（拒绝布尔/小数值）")
    return value


def verified_region_from_report(
    rep: dict, retouch_id: str, ctx: dict, cmyk_full: np.ndarray
) -> VerifiedRegion:
    """从修色报告推导出**唯一的已验证选区/偏移**。

    依据链：retouch_id（验收请求绑定值）== 报告 ``request`` 完整请求指纹的
    规范 JSON 之 SHA-256 —— ``request`` 才是用户原始请求（含未裁剪矩形）；
    报告 ``region`` 必须恰好是该请求矩形与当前原任务幅面的交点。任何分数/
    布尔坐标或偏移、与请求指纹不符的记录一律 :class:`AcceptanceError`，
    绝不做 ``int()`` 截断默许。
    """
    meta = ctx["meta"]
    req = rep.get("request")
    if not isinstance(req, dict):
        raise AcceptanceError("修色报告缺少完整请求指纹，拒绝验收")
    # 指纹绑定：报告 request 的原像哈希必须就是所绑 retouch_id，防止归档
    # 报告的请求/选区/约束被另一份记录替换而仍按 ID 误用。
    if hashlib.sha256(
        rt._canonical_json(req).encode("utf-8")
    ).hexdigest() != retouch_id:
        raise AcceptanceError(
            "修色报告请求指纹的 SHA-256 与所绑修色 ID 不符，拒绝验收"
        )
    if req.get("job_id") != meta["job_id"]:
        raise AcceptanceError("修色报告请求指纹绑定的原任务与所选任务不符，拒绝验收")
    if req.get("source_intent") != meta.get("source_intent"):
        raise AcceptanceError("修色报告请求指纹的渲染意图与原任务不符，拒绝验收")

    # 请求记录必须与 /api/retouch 同一严格口径：x/y/width/height/max_shift/
    # tac_limit 为非布尔整数且范围合法，目标 Lab/容差合法（缺失即拒绝）。
    try:
        req_validated = rt.validate_request({
            "job_id": meta["job_id"],
            "x": req.get("x"),
            "y": req.get("y"),
            "width": req.get("width"),
            "height": req.get("height"),
            "target_lab": req.get("target_lab"),
            "tolerance_de": req.get("tolerance_de"),
            "max_shift": req.get("max_shift"),
            "tac_limit": req.get("tac_limit"),
        })
    except rt.RetouchError as exc:
        raise AcceptanceError(f"修色报告请求指纹记录非法：{exc}")

    # 请求矩形与当前原任务幅面求交：贴边裁剪后的实际选区（唯一口径）。
    H, W = cmyk_full.shape[:2]
    try:
        vx, vy, vw, vh = rt.clip_region(cmyk_full, req_validated)
    except rt.RetouchError as exc:
        raise AcceptanceError(f"修色报告请求矩形超出原任务图像范围：{exc}")
    expected_region = {"x": int(vx), "y": int(vy),
                       "width": int(vw), "height": int(vh)}

    # region 逐字段核验。注意 Python 中 True == 1、1.0 == 1，字典相等比较
    # 无法识别布尔/分数坐标，因此先做严格类型检查：键恰好四个、值全为非布尔
    # int，再与求交结果精确比较——被改成 x=4.5/width=1.5（或 True）的归档
    # 报告在此直接 422，绝不 int() 截断默许。
    region = rep.get("region")
    region_keys = ("x", "y", "width", "height")
    if not isinstance(region, dict) or set(region) != set(region_keys):
        raise AcceptanceError(
            "修色报告记录的实际选区结构非法（需要且仅需要 x/y/width/height），拒绝验收"
        )
    region_strict = {k: _strict_int(region[k], f"region.{k}") for k in region_keys}
    if region_strict != expected_region:
        raise AcceptanceError(
            "修色报告记录的实际选区与请求矩形贴边裁剪后的选区不符，拒绝验收"
        )

    raw_offset = rep.get("best_offset_cmyk")
    if not isinstance(raw_offset, list) or len(raw_offset) != 4:
        raise AcceptanceError("修色报告缺少有效的统一 CMYK 偏移记录，拒绝验收")
    offset = tuple(_strict_int(v, "best_offset_cmyk") for v in raw_offset)

    # 偏移必须对已验证选区逐像素可行（字节范围/单通道上限/总墨量口径与
    # 修色搜索完全一致）；不可行偏移不可能产出被核验通过的成品。
    roi = cmyk_full[vy:vy + vh, vx:vx + vw]
    if not rt._offset_is_feasible(
        roi, offset, req_validated.max_shift, req_validated.tac_limit
    ):
        raise AcceptanceError(
            "修色报告记录的统一偏移对已验证选区不满足字节范围/单通道/总墨量"
            "约束，拒绝验收"
        )

    return VerifiedRegion(vx, vy, vw, vh, offset)


def verify_retouch_binding(
    retouch_id: str,
    retouch_base: Path,
    ctx: dict,
    cmyk_full: np.ndarray,
    press_bytes: bytes,
) -> tuple[dict, VerifiedRegion]:
    """独立核验“达标修色成品”与本任务/原印刷 ICC 的内容绑定。

    逐项核对：

    * 修色报告存在、结构完整、``reachable == true``、绑定的源任务就是本任务；
    * 报告 ``request`` 完整请求指纹的 SHA-256 == 所绑 retouch_id，请求整数
      语义合法；报告 ``region`` 恰为请求矩形与原任务幅面求交（贴边裁剪）后
      的实际选区、报告偏移为对该选区可行的非布尔整数四元组；
    * 报告记录的印刷 ICC SHA-256 == 当前原印刷 ICC 字节摘要；
    * ``retouched.tif`` 存在、可打开、CMYK、尺寸与原任务一致、嵌入的印刷
      ICC 字节与当前原印刷 ICC 完全相同；
    * 解码后逐像素核对（用**同一个已验证选区**）：选区**外**与原分色逐字节
      相同，选区**内**恰为“原分色 + 报告统一整数偏移”（夹到 0..255，与成品
      导出路径一致）。

    任一不符即拒绝验收，绝不在错误的成品上回算、也不用原任务值代替。
    返回 ``(修色报告 dict, 已验证选区)``——调用方的逐点内外判定必须复用
    返回的已验证选区，不得再从报告原值自行推导。
    """
    report_path = retouch_base / "report.json"
    tiff_path = retouch_base / "retouched.tif"
    if not report_path.exists():
        raise AcceptanceNotFound("修色报告不存在或已被删除")
    try:
        rep = json.loads(report_path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise AcceptanceError(f"修色报告不可读: {exc}")

    meta = ctx["meta"]
    if not isinstance(rep, dict):
        raise AcceptanceError("修色报告结构非法，拒绝验收")
    if rep.get("retouch_id") != retouch_id:
        raise AcceptanceError("修色报告记录的修色 ID 与所选修色不符，拒绝验收")
    if rep.get("source_job", {}).get("job_id") != meta["job_id"]:
        raise AcceptanceError("修色报告绑定的原任务与所选任务不符，拒绝验收")
    if rep.get("reachable") is not True:
        raise AcceptanceError("所选修色未达标（不可达），没有可用于印后验收的成品")
    press_summary = rep.get("profiles", {}).get("press", {})
    if press_summary.get("sha256") != hashlib.sha256(press_bytes).hexdigest():
        raise AcceptanceError(
            "修色报告绑定的印刷 ICC 摘要与原任务当前印刷 ICC 不符，拒绝验收"
        )

    # 唯一的已验证选区/偏移：指纹 + 幅面求交 + 整数/可行性严格核验。
    verified = verified_region_from_report(rep, retouch_id, ctx, cmyk_full)
    x, y, rw, rh = verified.x, verified.y, verified.width, verified.height
    off = np.asarray(verified.offset, dtype=np.int32)

    H, W = cmyk_full.shape[:2]

    if not tiff_path.is_file():
        raise AcceptanceError("达标修色成品 retouched.tif 缺失，拒绝验收")
    try:
        with Image.open(tiff_path) as im:
            im.load()
            if im.mode != "CMYK":
                raise AcceptanceError("修色成品不是 CMYK TIFF，拒绝验收")
            if im.size != (W, H):
                raise AcceptanceError("修色成品尺寸与原任务不一致，拒绝验收")
            embedded = im.info.get("icc_profile")
            tiff_pixels = np.asarray(im)
    except AcceptanceError:
        raise
    except Exception as exc:
        raise AcceptanceError(f"修色成品 TIFF 无法打开: {exc}")

    if embedded != press_bytes:
        raise AcceptanceError("修色成品嵌入的印刷 ICC 与原任务印刷 ICC 字节不符，拒绝验收")

    # 逐像素内容核验（与逐点预测共用 verified 选区）：选区外 == 原分色；
    # 选区内 == 原分色 + 统一偏移。
    expected = cmyk_full.copy()
    roi = expected[y:y + rh, x:x + rw]
    expected[y:y + rh, x:x + rw] = np.clip(
        roi.astype(np.int32) + off[None, None, :], 0, 255
    ).astype(np.uint8)
    if tiff_pixels.shape != expected.shape or not np.array_equal(tiff_pixels, expected):
        raise AcceptanceError(
            "修色成品像素与“原分色 + 报告统一偏移（选区外字节不变）”不符，拒绝验收"
        )
    return rep, verified


# ---------------------------------------------------------------------------
# 预测 Lab 回算与逐点 ΔE00
# ---------------------------------------------------------------------------

def predicted_labs(
    points: list[Measurement],
    ctx: dict,
    press_bytes: bytes,
    verified_region: VerifiedRegion | None,
    verified_tiff: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    """按策略返回每个测点的预测 Lab 与来源标记。

    无修色：全部取原任务 ``lab_press.npy``（``original_job_prediction``）。
    绑定修色：**已验证选区**内的点从已核验成品 TIFF 的 CMYK 经原印刷 ICC、
    原任务意图回算（``verified_retouched_product``）；选区外保持原任务预测
    值。选区与 TIFF 像素核验共用同一个 :class:`VerifiedRegion`，因此“内外
    判定”与“像素在成品中是否被改写”永远一致。
    """
    meta = ctx["meta"]
    intent = meta.get(PRESS_INTENT_FIELD, "relative_colorimetric")
    if intent not in lcms_native.INTENTS:
        raise AcceptanceError(f"原任务渲染意图无效: {intent}")

    n = len(points)
    lab_press = ctx["lab_press"]
    predicted = np.zeros((n, 3), dtype=np.float64)
    sources = ["original_job_prediction"] * n
    inside = np.zeros(n, dtype=bool)

    if verified_region is not None:
        x0, y0 = verified_region.x, verified_region.y
        x1, y1 = x0 + verified_region.width, y0 + verified_region.height
        for i, p in enumerate(points):
            inside[i] = x0 <= p.x < x1 and y0 <= p.y < y1

        # 选区内：从已核验成品取 CMYK，按原印刷 ICC + 原任务意图回算 Lab。
        idx_in = np.nonzero(inside)[0]
        if idx_in.size:
            ys = np.array([points[i].y for i in idx_in], dtype=np.intp)
            xs = np.array([points[i].x for i in idx_in], dtype=np.intp)
            cmyk_sel = np.ascontiguousarray(verified_tiff[ys, xs].reshape(-1, 4))
            with lcms_native.lab_transform("CMYK", press_bytes, intent) as xf:
                lab_sel = xf(cmyk_sel)
            predicted[idx_in] = lab_sel
            for i in idx_in:
                sources[int(i)] = "verified_retouched_product"

        # 选区外：原任务逐像素预测值（绝不改写）。
        idx_out = np.nonzero(~inside)[0]
        if idx_out.size:
            ys = np.array([points[i].y for i in idx_out], dtype=np.intp)
            xs = np.array([points[i].x for i in idx_out], dtype=np.intp)
            predicted[idx_out] = np.asarray(lab_press[ys, xs], dtype=np.float64)
    else:
        ys = np.array([p.y for p in points], dtype=np.intp)
        xs = np.array([p.x for p in points], dtype=np.intp)
        predicted[:] = np.asarray(lab_press[ys, xs], dtype=np.float64)

    return predicted, sources


# ---------------------------------------------------------------------------
# 身份指纹（规范 JSON，浮点按最短往返表示）
#
# 指纹同时包含**原始 CSV 字节 SHA-256** 与规范后的测点序列：解析出相同测点、
# 但原始字节不同的两份 CSV（仅表头别名/空白/BOM/换行不同）会得到**不同**
# acceptance_id，各自寻址、各自保存原件与报告，避免同身份下原件错配。
# ---------------------------------------------------------------------------

def _canon_float(v: float) -> float:
    v = float(v)
    return 0.0 if v == 0.0 else v


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_fingerprint(meta: dict, points: list[Measurement],
                        limit_de: float, retouch_id: str | None,
                        press_digest: str, csv_sha256: str) -> dict:
    return {
        "kind": "postpress-color-acceptance",
        "job_id": meta["job_id"],
        "source_intent": meta.get(PRESS_INTENT_FIELD, ""),
        "retouch_id": retouch_id,
        "press_profile_sha256": press_digest,
        # 原始 CSV 字节摘要参与身份：同一规范测点序列的不同原件分别寻址。
        "measurements_csv_sha256": csv_sha256,
        "limit_de00": _canon_float(limit_de),
        "points": [
            {"x": int(p.x), "y": int(p.y),
             "L": _canon_float(p.L), "a": _canon_float(p.a), "b": _canon_float(p.b)}
            for p in points
        ],
    }


def compute_acceptance_id(fingerprint: dict) -> str:
    return hashlib.sha256(_canonical_json(fingerprint).encode("utf-8")).hexdigest()


def read_stored_csv(csv_path: Path) -> bytes | None:
    """读取归档测点原件；不存在或不可读时返回 None。"""
    try:
        return csv_path.read_bytes()
    except OSError:
        return None


def csv_bytes_match_report(report: dict, blob: bytes) -> bool:
    """给定字节的 SHA-256 是否与报告固化的原件摘要严格一致。"""
    expected = report.get("measurements_csv_sha256")
    return (
        isinstance(expected, str)
        and len(expected) == 64
        and hashlib.sha256(blob).hexdigest() == expected
    )


def stored_csv_matches_report(report: dict, csv_path: Path) -> bool:
    """归档 measurements.csv 的 SHA-256 是否与报告固化摘要严格一致。"""
    blob = read_stored_csv(csv_path)
    return blob is not None and csv_bytes_match_report(report, blob)


# ---------------------------------------------------------------------------
# 验收主流程
# ---------------------------------------------------------------------------

def evaluate(
    points: list[Measurement],
    limit_de: float,
    ctx: dict,
    press_bytes: bytes,
    press_summary: dict,
    retouch_id: str | None,
    retouch_report: dict | None,
    verified_region: VerifiedRegion | None,
    verified_tiff: np.ndarray | None,
) -> dict:
    """逐点回算预测 Lab、计算 ΔE00 并给出整张判定。"""
    meta = ctx["meta"]
    predicted, sources = predicted_labs(
        points, ctx, press_bytes, verified_region, verified_tiff
    )
    measured = np.array([[p.L, p.a, p.b] for p in points], dtype=np.float64)
    n = len(points)
    de = colorscience.ciede2000(
        measured.reshape(n, 1, 3), predicted.reshape(n, 1, 3)
    ).reshape(n)

    # 任一点严格大于上限即整张不合格；相等视为合格。
    over = de > float(limit_de)
    point_records = []
    over_points = []
    for i, p in enumerate(points):
        rec = {
            "seq": i + 1,
            "x": int(p.x),
            "y": int(p.y),
            "measured_lab": [round(float(v), 4) for v in measured[i]],
            "predicted_lab": [round(float(v), 4) for v in predicted[i]],
            "de00": round(float(de[i]), 4),
            "prediction_source": sources[i],
            "over_limit": bool(over[i]),
        }
        point_records.append(rec)
        if over[i]:
            over_points.append({"seq": i + 1, "x": int(p.x), "y": int(p.y),
                                "de00": round(float(de[i]), 4)})

    passed = bool(not over.any())
    return {
        "passed": passed,
        "limit_de00": float(limit_de),
        "point_count": n,
        "over_limit_count": int(over.sum()),
        "over_limit_points": over_points,
        "stats": {
            "de00_mean": round(float(de.mean()), 4),
            "de00_max": round(float(de.max()), 4),
            "de00_min": round(float(de.min()), 4),
        },
        "points": point_records,
        "prediction_method": {
            "press_intent": meta.get(PRESS_INTENT_FIELD, "relative_colorimetric"),
            "strategy": (
                "verified_retouched_product_inside__original_job_outside"
                if retouch_id is not None
                else "original_job_lab_press"
            ),
            "note": (
                "修色选区内测点从已核验成品 TIFF 的 CMYK 按原印刷 ICC 与原任务"
                "意图回算为 Lab（成品经逐像素核验恰为原分色+统一偏移，与按原分色"
                "与偏移重建等价）；选区外保持原任务 lab_press.npy 预测值。"
                if retouch_id is not None
                else "逐点预测 Lab 取原任务 lab_press.npy（分色 CMYK 经原印刷 ICC "
                     "与原任务意图回算的逐像素预测值）。"
            ),
        },
        "press_profile": press_summary,
    }


def build_report(
    acceptance_id: str,
    fingerprint: dict,
    ctx: dict,
    points: list[Measurement],
    limit_de: float,
    retouch_id: str | None,
    retouch_report: dict | None,
    verified_region: VerifiedRegion | None,
    evaluation: dict,
    csv_sha256: str,
) -> dict:
    meta = ctx["meta"]
    report = {
        "acceptance_id": acceptance_id,
        "kind": "postpress_color_acceptance",
        "request": fingerprint,
        "source_job": {
            "job_id": meta["job_id"],
            "image_sha256": meta.get("image_sha256"),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "image_mode": meta.get("mode"),
            "source_intent": meta.get("source_intent"),
            "proof_intent": meta.get("proof_intent"),
            "threshold_delta_e": meta.get("threshold"),
        },
        "retouch": (
            {
                "retouch_id": retouch_id,
                # 写入**已验证选区/偏移**（与像素核验、逐点判定同一来源），
                # 不照抄可能被改写的报告原值。
                "region": verified_region.rect,
                "best_offset_cmyk": list(verified_region.offset),
                "reachable": True,
            }
            if retouch_id is not None
            else None
        ),
        "profiles": {
            "press": evaluation["press_profile"],
        },
        "rule": {
            "color_difference": "CIEDE2000 (D50)",
            "pass_condition": "所有测点 ΔE00 <= 上限；任一点严格大于上限即整张不合格",
            "duplicate_coordinates": "同坐标复测逐条保留、不去重",
            "limit_de00": float(limit_de),
        },
        "measurements_csv_sha256": csv_sha256,
        "passed": evaluation["passed"],
        "point_count": evaluation["point_count"],
        "over_limit_count": evaluation["over_limit_count"],
        "over_limit_points": evaluation["over_limit_points"],
        "stats": evaluation["stats"],
        "prediction_method": evaluation["prediction_method"],
        "points": evaluation["points"],
        "artifacts": {"report": "report.json", "measurements": "measurements.csv"},
        "note": (
            "印后测色验收：预测 Lab 均按原印刷 ICC 与原任务意图回算；绑定修色时"
            "仅修色选区内从已核验成品回算，选区外保持原任务预测值；绝不以原任务"
            "值代替修色测点。修色不可达、报告/成品绑定不符、ICC 缺失或摘要不符、"
            "测点越界或 CSV 非法时一律拒绝验收。"
        ),
    }
    return report


def verify_report_binding(
    report: object,
    acceptance_id: str,
    fingerprint: dict,
    ctx: dict,
    points: list[Measurement],
    limit_de: float,
    press_bytes: bytes,
    press_summary: dict,
    retouch_id: str | None,
    retouch_report: dict | None,
    verified_region: VerifiedRegion | None,
    verified_tiff: np.ndarray | None,
    csv_sha256: str,
) -> bool:
    """命中缓存时逐项核对报告与**本次请求**的绑定及全部逐点结果。

    指纹已包含原始 CSV 字节摘要，故这里核对 ``request == fingerprint`` 即同时
    确认“规范测点序列 + 原件摘要”与本次提交一致；再显式核对报告顶层记录的
    CSV 摘要。注意：本函数只核对报告**内容**，不触碰磁盘上的 measurements.csv
    ——归档原件的存在性与字节摘要由调用方另行核对/修复（见 main 流程）。

    任一不符返回 False（调用方据此拒绝验收，而不是用原任务值代替或静默重算）。
    """
    try:
        if not isinstance(report, dict):
            return False
        if report.get("acceptance_id") != acceptance_id:
            return False
        if report.get("request") != fingerprint:
            return False
        if report.get("measurements_csv_sha256") != csv_sha256:
            return False
        if report.get("profiles", {}).get("press", {}).get("sha256") != \
                hashlib.sha256(press_bytes).hexdigest():
            return False
        expected = evaluate(
            points, limit_de, ctx, press_bytes, press_summary,
            retouch_id, retouch_report, verified_region, verified_tiff,
        )
        if report.get("passed") != expected["passed"]:
            return False
        if report.get("point_count") != expected["point_count"]:
            return False
        if report.get("over_limit_count") != expected["over_limit_count"]:
            return False
        if report.get("over_limit_points") != expected["over_limit_points"]:
            return False
        for key in ("de00_mean", "de00_max", "de00_min"):
            if abs(float(report["stats"][key]) - float(expected["stats"][key])) > 5e-5:
                return False
        old, new = report.get("points"), expected["points"]
        if not isinstance(old, list) or len(old) != len(new):
            return False
        for a, b in zip(old, new):
            if a["x"] != b["x"] or a["y"] != b["y"]:
                return False
            if a["prediction_source"] != b["prediction_source"] or \
                    a["over_limit"] != b["over_limit"]:
                return False
            for k in ("measured_lab", "predicted_lab"):
                if not (isinstance(a.get(k), list) and len(a[k]) == 3):
                    return False
                if any(abs(float(x) - float(y)) > 5e-5
                       for x, y in zip(a[k], b[k])):
                    return False
            if abs(float(a["de00"]) - float(b["de00"])) > 5e-5:
                return False
        if retouch_id is not None:
            rr = report.get("retouch")
            if not isinstance(rr, dict) or rr.get("retouch_id") != retouch_id:
                return False
            # 与**已验证选区**（请求矩形贴边裁剪后的实际交点）逐项一致，
            # 不直接相信缓存报告里的 region/offset 原值。注意 True==1、
            # 4.0==4：类型必须也是非布尔 int，防止归档报告被改成等价值。
            cached_region = rr.get("region")
            if not isinstance(cached_region, dict) or \
                    set(cached_region) != set(verified_region.rect):
                return False
            for k, v in verified_region.rect.items():
                cv = cached_region.get(k)
                if isinstance(cv, bool) or not isinstance(cv, int) or cv != v:
                    return False
            cached_off = rr.get("best_offset_cmyk")
            if not isinstance(cached_off, list) or len(cached_off) != 4:
                return False
            for cv, v in zip(cached_off, verified_region.offset):
                if isinstance(cv, bool) or not isinstance(cv, int) or cv != v:
                    return False
            if rr.get("reachable") is not True:
                return False
        elif report.get("retouch") is not None:
            return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def list_retouches_for_job(retouch_dir: Path, job_id: str) -> list[dict]:
    """列出某任务下可用于印后验收的**达标**修色成品（扫描内容寻址目录）。

    列表只做展示性读取：选区/偏移记录不是严格整数结构（被改成浮点/布尔或
    结构损坏）的报告一律跳过——真正的绑定核验在提交验收时逐项执行。
    """
    out: list[dict] = []
    if not retouch_dir.exists():
        return out
    for d in sorted(retouch_dir.iterdir()):
        rp = d / "report.json"
        if not d.is_dir() or not rp.is_file():
            continue
        try:
            rep = json.loads(rp.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(rep, dict):
            continue
        if rep.get("source_job", {}).get("job_id") != job_id:
            continue
        if rep.get("reachable") is not True or not (d / "retouched.tif").is_file():
            # 不可达修色没有成品，不能用于印后验收。
            continue
        region = rep.get("region")
        offset = rep.get("best_offset_cmyk")
        if not (isinstance(region, dict)
                and all(isinstance(region.get(k), int)
                        and not isinstance(region.get(k), bool)
                        for k in ("x", "y", "width", "height"))):
            continue
        if not (isinstance(offset, list) and len(offset) == 4
                and all(isinstance(v, int) and not isinstance(v, bool)
                        for v in offset)):
            continue
        out.append({
            "retouch_id": rep.get("retouch_id") or d.name,
            "region": region,
            "best_offset_cmyk": offset,
            "reachable": True,
        })
    return out
