// 印后测色验收视图（原生 JavaScript）
"use strict";

const acState = {
  jobs: [],
  jobId: null,
  jobReport: null,
  retouches: [],
  result: null,
};

const ac$ = (id) => document.getElementById(id);

async function acApi(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      if (j && j.detail) msg = j.detail;
    } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

function acShowError(msg) {
  const box = ac$("acError");
  if (!msg) { box.hidden = true; box.textContent = ""; return; }
  box.hidden = false;
  box.textContent = msg;
}

async function acLoadJobs(selectId) {
  acState.jobs = await acApi("/api/jobs");
  const sel = ac$("acJobSelect");
  sel.innerHTML = "";
  if (!acState.jobs.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "（尚无已保存任务，请先在“软打样”页运行一次）";
    sel.appendChild(opt);
    acState.jobId = null;
    return;
  }
  for (const j of acState.jobs) {
    const opt = document.createElement("option");
    opt.value = j.job_id;
    opt.textContent =
      `${j.job_id.slice(0, 12)}… ${j.width}×${j.height} ${j.image_mode} · ` +
      `${new Date(j.created_at + "Z").toLocaleString()}`;
    sel.appendChild(opt);
  }
  if (selectId) sel.value = selectId;
}

async function acLoadRetouches() {
  const sel = ac$("acRetouchSelect");
  sel.innerHTML = "";
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "不使用修色（直接按原任务 lab_press 预测）";
  sel.appendChild(none);
  acState.retouches = [];
  if (!acState.jobId) return;
  acState.retouches = await acApi(`/api/jobs/${acState.jobId}/retouches`);
  for (const r of acState.retouches) {
    const opt = document.createElement("option");
    opt.value = r.retouch_id;
    const rg = r.region;
    opt.textContent =
      `达标修色 ${r.retouch_id.slice(0, 12)}… 选区(${rg.x},${rg.y},` +
      `${rg.width}×${rg.height}) 偏移(${r.best_offset_cmyk.join(",")})`;
    sel.appendChild(opt);
  }
}

async function acSelectJob() {
  const id = ac$("acJobSelect").value;
  acState.jobId = id || null;
  ac$("acResultPanel").hidden = true;
  if (!id) {
    acState.jobReport = null;
    ac$("acJobMeta").textContent = "";
    await acLoadRetouches();
    return;
  }
  acState.jobReport = await acApi(`/api/jobs/${id}`);
  const p = acState.jobReport.profiles.press;
  ac$("acJobMeta").innerHTML =
    `任务 <code>${id.slice(0, 12)}…</code> · ` +
    `${acState.jobReport.image.width}×${acState.jobReport.image.height} · ` +
    `印刷 ICC <b>${p.name}</b> · SHA-256 <code>${p.sha256.slice(0, 16)}…</code> · ` +
    `意图 ${acState.jobReport.parameters.source_intent_label}`;
  ac$("acViewImage").src = `/api/jobs/${id}/images/softproof.png?cache=${id}`;
  await acLoadRetouches();
}

// ---- 超限测点页面标注 ------------------------------------------------------
const acImg = ac$("acViewImage"), acCanvas = ac$("acMarkCanvas");

function acRoiRect() {
  if (!acImg || !acImg.naturalWidth) return;
  const wrap = ac$("acCanvasWrap");
  const wRect = wrap.getBoundingClientRect();
  const r = acImg.getBoundingClientRect();
  acCanvas.style.position = "absolute";
  acCanvas.style.left = (r.left - wRect.left + wrap.scrollLeft) + "px";
  acCanvas.style.top = (r.top - wRect.top + wrap.scrollTop) + "px";
  acCanvas.style.width = r.width + "px";
  acCanvas.style.height = r.height + "px";
  acCanvas.width = Math.round(r.width);
  acCanvas.height = Math.round(r.height);
}
acImg.addEventListener("load", acRoiRect);
window.addEventListener("resize", acRoiRect);

function acDrawMarks() {
  acRoiRect();
  const ctx = acCanvas.getContext("2d");
  ctx.clearRect(0, 0, acCanvas.width, acCanvas.height);
  const res = acState.result;
  if (!res || !acImg.naturalWidth) return;
  const sx = acCanvas.width / acImg.naturalWidth;
  const sy = acCanvas.height / acImg.naturalHeight;
  for (const pt of res.points) {
    const cx = pt.x * sx, cy = pt.y * sy;
    const over = pt.over_limit;
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(4, 9 * sx), 0, Math.PI * 2);
    ctx.fillStyle = over ? "rgba(255,0,0,.28)" : "rgba(40,180,90,.14)";
    ctx.fill();
    ctx.lineWidth = over ? 2.2 : 1.0;
    ctx.strokeStyle = over ? "#ff1a1a" : "rgba(40,160,80,.8)";
    ctx.stroke();
    if (over) {
      // 红框 + 十字，保证超限坐标在页面上被明确标出。
      ctx.beginPath();
      ctx.moveTo(cx - 6 * sx, cy - 6 * sy);
      ctx.lineTo(cx + 6 * sx, cy + 6 * sy);
      ctx.moveTo(cx + 6 * sx, cy - 6 * sy);
      ctx.lineTo(cx - 6 * sx, cy + 6 * sy);
      ctx.stroke();
    }
  }
}

// ---- 执行验收 --------------------------------------------------------------
ac$("acRefreshJobs").addEventListener("click", () =>
  acLoadJobs(ac$("acJobSelect").value)
    .then(acSelectJob)
    .catch((e) => acShowError(e.message))
);
ac$("acJobSelect").addEventListener("change", () =>
  acSelectJob().catch((e) => acShowError(e.message))
);
ac$("acCsvFile").addEventListener("change", () => {
  const f = ac$("acCsvFile").files[0];
  ac$("acCsvInfo").textContent = f ? `${f.name} (${f.size} 字节)` : "";
});

ac$("acRunBtn").addEventListener("click", async () => {
  acShowError("");
  if (!acState.jobId) return acShowError("请先选择一个已保存任务");
  const file = ac$("acCsvFile").files[0];
  if (!file) return acShowError("请先选择测点 CSV 文件");
  const limit = parseFloat(ac$("acLimit").value);
  if (!Number.isFinite(limit) || limit < 0)
    return acShowError("ΔE00 上限必须是非负有限数值");
  const rid = ac$("acRetouchSelect").value || "";

  const fd = new FormData();
  fd.append("job_id", acState.jobId);
  fd.append("limit_de", String(limit));
  fd.append("file", file);
  if (rid) fd.append("retouch_id", rid);

  ac$("acRunBtn").disabled = true;
  ac$("acBusy").hidden = false;
  try {
    acState.result = await acApi("/api/acceptance", {
      method: "POST", body: fd,
    });
    acRenderResult();
  } catch (err) {
    ac$("acResultPanel").hidden = true;
    acShowError(err.message);
  } finally {
    ac$("acRunBtn").disabled = false;
    ac$("acBusy").hidden = true;
  }
});

function acLab(v) {
  return Array.isArray(v)
    ? `(${v.map((x) => Number(x).toFixed(2)).join(", ")})`
    : "—";
}

function acRenderResult() {
  const r = acState.result;
  ac$("acResultPanel").hidden = false;
  const status = ac$("acStatus");
  if (r.passed) {
    status.className = "rt-status reachable";
    status.textContent =
      `合格：${r.point_count} 个测点全部满足 ΔE00 ≤ 上限 ${r.limit_de00}。`;
  } else {
    status.className = "rt-status unreachable";
    status.textContent =
      `不合格：${r.point_count} 个测点中有 ${r.over_limit_count} 个 ` +
      `ΔE00 严格大于上限 ${r.limit_de00}，整张验收不通过。`;
  }

  ac$("acStats").innerHTML =
    `<span>测点数 <b>${r.point_count}</b></span>` +
    `<span>ΔE00 均值 <b>${r.stats.de00_mean}</b></span>` +
    `<span>ΔE00 最大 <b>${r.stats.de00_max}</b></span>` +
    `<span>ΔE00 最小 <b>${r.stats.de00_min}</b></span>` +
    `<span>超限点数 <b>${r.over_limit_count}</b></span>` +
    `<span>验收 ID <code>${r.acceptance_id.slice(0, 12)}…</code></span>`;

  const overBox = ac$("acOverBox");
  if (r.over_limit_points.length) {
    overBox.innerHTML =
      `<h3>超限坐标（${r.over_limit_count} 个）</h3>` +
      r.over_limit_points.map(
        (p) => `<span class="ac-over-chip">(x=${p.x}, y=${p.y}) ΔE00=${p.de00}</span>`
      ).join(" ");
  } else {
    overBox.innerHTML = "<h3>无超限测点。</h3>";
  }

  const tbody = ac$("acPointsTable").querySelector("tbody");
  tbody.innerHTML = "";
  for (const p of r.points) {
    const tr = document.createElement("tr");
    tr.className = p.over_limit ? "ac-row-over" : "";
    tr.innerHTML =
      `<td>${p.seq}</td><td>${p.x}</td><td>${p.y}</td>` +
      `<td>${acLab(p.measured_lab)}</td><td>${acLab(p.predicted_lab)}</td>` +
      `<td>${p.de00}</td>` +
      `<td class="ac-src">${p.prediction_source === "verified_retouched_product"
        ? "已核验修色成品回算" : "原任务预测值"}</td>` +
      `<td>${p.over_limit ? '<b class="ac-bad">超限</b>' : "通过"}</td>`;
    tbody.appendChild(tr);
  }

  ac$("acExpReport").href = r.report_url;
  ac$("acExpReport").download = `${r.acceptance_id.slice(0, 12)}-acceptance.json`;
  ac$("acExpCsv").href = `/api/acceptance/${r.acceptance_id}/measurements.csv`;
  ac$("acReportJson").textContent = JSON.stringify(r.report, null, 2);

  // 等软打样图就绪后再绘制标注。
  if (acImg.complete && acImg.naturalWidth) acDrawMarks();
  else acImg.onload = acDrawMarks;
  setTimeout(acDrawMarks, 0);
}

// ---- 视图切换钩子（app.js 的导航处理调用） -------------------------------
async function refreshAcceptanceView() {
  try {
    if (!acState.jobs.length) {
      await acLoadJobs();
      if (acState.jobId === null && acState.jobs.length) await acSelectJob();
    }
    setTimeout(acRoiRect, 0);
  } catch (e) {
    acShowError("加载已保存任务失败: " + e.message);
  }
}
