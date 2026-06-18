const $ = (selector) => document.querySelector(selector);

const statusText = $("#statusText");
const stepText = $("#stepText");
const progressBar = $("#progressBar");
const logBox = $("#logBox");
const metricCount = $("#metricCount");
const metricAccuracy = $("#metricAccuracy");
const metricReid = $("#metricReid");
const summaryGrid = $("#summaryGrid");
const artifactList = $("#artifactList");
const runButton = $("#runButton");

let polling = null;

async function getJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || data.message || "Error de servidor");
  }
  return data;
}

function labelStatus(status) {
  const labels = {
    idle: "Esperando",
    queued: "En cola",
    running: "Procesando",
    completed: "Finalizado",
    failed: "Error",
  };
  return labels[status] || status;
}

function updateMetrics(summary) {
  if (!summary || Object.keys(summary).length === 0) {
    metricCount.textContent = "-";
    metricAccuracy.textContent = "-";
    metricReid.textContent = "-";
    summaryGrid.innerHTML = "";
    return;
  }

  metricCount.textContent = `${summary.estimated_total_cows ?? "-"} vacas`;
  metricAccuracy.textContent = `${summary.count_accuracy_percent ?? "-"}%`;
  metricReid.textContent = (summary.known_found || []).join(", ") || "-";

  const cards = [
    ["Frames", summary.processed_frames],
    ["Conteo estimado", summary.estimated_total_cows],
    ["Referencia", summary.expected_total_cows],
    ["Error", summary.count_error],
    ["Accuracy", `${summary.count_accuracy_percent}%`],
    ["ID switches", summary.known_id_switches_by_design],
    ["Re-ID encontradas", (summary.known_found || []).join(", ")],
    ["Re-ID faltantes", (summary.known_missing || []).join(", ") || "ninguna"],
    ["Tracking auditado", summary.locked_track_audit_ok ? "OK" : "Revisar"],
  ];

  summaryGrid.innerHTML = cards
    .map(([label, value]) => `<div><span>${label}</span><strong>${value ?? "-"}</strong></div>`)
    .join("");
}

function updateArtifacts(artifacts) {
  if (!artifacts || Object.keys(artifacts).length === 0) {
    artifactList.innerHTML = "";
    return;
  }
  const rows = [
    ["Video renderizado", artifacts.video_path],
    ["Reporte JSON", artifacts.report_path],
    ["Contact sheet", artifacts.contact_sheet_path],
  ].filter(([, value]) => value);

  artifactList.innerHTML = rows
    .map(([label, value]) => `<code><strong>${label}</strong><br>${value}</code>`)
    .join("");
}

async function refreshStatus() {
  const state = await getJson("/api/status");
  statusText.textContent = labelStatus(state.status);
  stepText.textContent = state.step || "";
  progressBar.style.width = `${state.progress || 0}%`;
  logBox.textContent = (state.logs || []).join("\n") || "Los eventos del pipeline aparecerán acá.";
  logBox.scrollTop = logBox.scrollHeight;
  updateMetrics(state.summary);
  updateArtifacts(state.artifacts);
  runButton.disabled = state.status === "running" || state.status === "queued";
  runButton.textContent = runButton.disabled ? "Procesando..." : "Ejecutar CowTrack";
  if (state.status !== "running" && state.status !== "queued" && polling) {
    clearInterval(polling);
    polling = null;
  }
}

async function loadConfig() {
  const config = await getJson("/api/config");
  $("#videoPath").value = config.default_video;
  $("#resultDir").value = config.default_result_dir;
  if (config.telegram_ready) {
    $("#telegramEnabled").checked = true;
  }
}

$("#runForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  if (!$("#telegramEnabled").checked) {
    form.set("telegram_enabled", "false");
  }
  try {
    await getJson("/api/run", {
      method: "POST",
      body: form,
    });
    await refreshStatus();
    polling = setInterval(refreshStatus, 1600);
  } catch (error) {
    alert(error.message);
  }
});

$("#refreshButton").addEventListener("click", refreshStatus);

$("#telegramTest").addEventListener("click", async () => {
  const form = new FormData($("#runForm"));
  try {
    const data = await getJson("/api/telegram/test", {
      method: "POST",
      body: form,
    });
    alert(data.message || "Telegram configurado");
  } catch (error) {
    alert(error.message);
  }
});

loadConfig()
  .then(refreshStatus)
  .catch((error) => {
    logBox.textContent = error.message;
  });
