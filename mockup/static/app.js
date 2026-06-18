const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

let dashboardData = { catalog: [], reports: [], state: {} };
let statusTimer = null;
let currentUser = null;
let metricMode = "general";

async function api(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", ...options });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || data.message || "Error");
  return data;
}

function formData(form) { return new FormData(form); }
function todaySlug() { return new Date().toISOString().slice(0, 10); }

function showMessage(title, text, kicker = "CowTrack") {
  $("#messageKicker").textContent = kicker;
  $("#messageTitle").textContent = title;
  $("#messageText").textContent = text;
  $("#messageModal").classList.remove("hidden");
}

function hideMessage() { $("#messageModal").classList.add("hidden"); }
function openLogin() { $("#loginModal").classList.remove("hidden"); }
function closeLogin() { $("#loginModal").classList.add("hidden"); }
function openRegister() { closeLogin(); $("#registerModal").classList.remove("hidden"); }
function closeRegister() { $("#registerModal").classList.add("hidden"); }
function openTelegram() { $("#telegramModal").classList.remove("hidden"); }
function closeTelegram() { $("#telegramModal").classList.add("hidden"); }

function fileLink(path) {
  if (!path) return "";
  const mockupPrefix = "/Volumes/T7/cow-tracker-mvp/mockup/";
  const rootPrefix = "/Volumes/T7/cow-tracker-mvp/";
  if (path.startsWith(mockupPrefix)) return `/mockup-files/${path.slice(mockupPrefix.length)}`;
  if (path.startsWith(rootPrefix)) return `/cowtrack-files/${path.slice(rootPrefix.length)}`;
  return "";
}

function latestReport() {
  return dashboardData.reports[0] || {};
}

function latestValidReport() {
  return latestReport();
}

function reportAccuracy(report) {
  return Number(report.count_accuracy_percent || 0);
}

function metric(label, value, detail = "") {
  return `<article class="kpi-card"><span>${label}</span><strong>${value ?? "-"}</strong>${detail ? `<small>${detail}</small>` : ""}</article>`;
}

function donut(percent, label) {
  const value = Math.max(0, Math.min(100, Number(percent || 0)));
  return `<div class="donut" style="--value:${value}"><strong>${value.toFixed(value % 1 ? 1 : 0)}%</strong><span>${label}</span></div>`;
}

function barChart(items) {
  const max = Math.max(1, ...items.map((item) => Number(item.value || 0)));
  return `<div class="bar-chart">${items.map((item) => `
    <div class="bar-row">
      <span>${item.label}</span>
      <div><i style="width:${Math.max(4, (Number(item.value || 0) / max) * 100)}%"></i></div>
      <strong>${item.value}</strong>
    </div>
  `).join("")}</div>`;
}

function setPublicActive(id) {
  $$(".public-nav a").forEach((link) => link.classList.toggle("active", link.dataset.go === id));
}

function showPublicPage(id = "home") {
  document.body.classList.remove("dashboard-mode");
  $("#publicApp").classList.remove("hidden");
  $("#dashboardApp").classList.add("hidden");
  $("#dashboardOpen").classList.toggle("hidden", !currentUser);
  $("#loginOpen").classList.toggle("hidden", Boolean(currentUser));
  $$(".page-section").forEach((section) => section.classList.toggle("hidden", section.id !== id));
  setPublicActive(id);
  closeIntroAndOffer();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function openHashPage() {
  const id = (window.location.hash || "#home").replace("#", "");
  if (document.body.classList.contains("dashboard-mode") && id.startsWith("dashboard-")) return;
  if (document.getElementById(id)?.classList.contains("page-section")) showPublicPage(id);
}

function showDashboard(view = "overview") {
  document.body.classList.add("dashboard-mode");
  $("#publicApp").classList.add("hidden");
  $("#dashboardApp").classList.remove("hidden");
  $("#dashboardOpen").classList.add("hidden");
  $("#loginOpen").classList.add("hidden");
  closeIntroAndOffer();
  setView(view);
  ensureStatusPolling();
  window.history.replaceState(null, "", `#dashboard-${view}`);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function setView(view) {
  $$(".dash-view").forEach((el) => el.classList.add("hidden"));
  $(`#view-${view}`)?.classList.remove("hidden");
  $$(".dash-sidebar nav button").forEach((btn) => btn.classList.toggle("active", btn.dataset.view === view));
  if (document.body.classList.contains("dashboard-mode")) window.history.replaceState(null, "", `#dashboard-${view}`);
  if (dashboardData.state?.status === "running" || dashboardData.state?.status === "queued") updatePersistentProgress(dashboardData.state);
}

function closeIntroAndOffer() {
  $("#introVideo").classList.add("hidden");
  $("#launchOffer").classList.add("hidden");
  $("#adVideo").pause();
}

function maybeShowIntro() {
  const isHome = !window.location.hash || window.location.hash === "#home";
  if (currentUser || !isHome) return;
  $("#introVideo").classList.remove("hidden");
  $("#adVideo").currentTime = 0;
  $("#adVideo").play().catch(() => {});
}

function showOfferOnce() {
  if (currentUser) return;
  $("#launchOffer").classList.remove("hidden");
}

function isRunningState(state = dashboardData.state || {}) {
  return state.status === "running" || state.status === "queued";
}

function updatePersistentProgress(state = dashboardData.state || {}) {
  const running = isRunningState(state);
  const existing = $("#persistentProgress");
  if (!running) {
    existing?.remove();
    return;
  }
  const html = `
    <section id="persistentProgress" class="persistent-progress">
      <div>
        <strong>Conteo CowTrack en proceso</strong>
        <span>${state.step || "Procesando video"} · ${state.progress || 1}%</span>
      </div>
      <div class="mini-progress"><i style="width:${state.progress || 1}%"></i></div>
    </section>
  `;
  if (existing) existing.outerHTML = html;
  else $(".dash-top")?.insertAdjacentHTML("afterend", html);
}

function ensureStatusPolling() {
  if (statusTimer || !isRunningState(dashboardData.state)) return;
  statusTimer = setInterval(updateStatus, 1500);
}

function renderOverview() {
  const latest = latestValidReport();
  const running = isRunningState(dashboardData.state);
  $("#view-overview").innerHTML = `
    ${running ? `<section class="status-banner"><strong>Reporte en curso</strong><span>${dashboardData.state.step || "Procesando video"} · ${dashboardData.state.progress || 0}%</span></section>` : ""}
    <div class="analytics-grid">
      ${metric("Conteo consolidado", latest.estimated_total_cows ? `${latest.estimated_total_cows} vacas` : "Sin datos", `Referencia: ${latest.expected_total_cows || "-"} vacas`)}
      ${metric("Vacas catalogadas", dashboardData.catalog.length, "Identidades guardadas")}
      ${metric("Reidentificadas", latest.reidentified_count ?? (latest.reidentified_cows || []).length ?? 0, (latest.reidentified_cows || []).join(", ") || "Sin localizaciones")}
      ${metric("Reportes históricos", dashboardData.reports.length, "Historial operativo")}
    </div>
    <section class="dashboard-main-grid">
      <article class="panel executive-card">
        <span class="eyebrow">Resumen del último reporte</span>
        <h2>${latest.title || "Sin reportes"}</h2>
        <p>Estado: <strong>${latest.status_label || "-"}</strong>. El sistema compara el conteo consolidado contra la referencia ingresada y separa vacas catalogadas de vacas no catalogadas.</p>
        <div class="mini-metrics">
          ${donut(reportAccuracy(latest), "confiabilidad de conteo")}
          ${donut(((latest.reidentified_count || 0) / Math.max(1, dashboardData.catalog.length)) * 100, "Re-ID catálogo")}
        </div>
      </article>
      <article class="panel">
        <h2>Composición del rodeo</h2>
        ${barChart([
          { label: "Catalogadas", value: latest.reidentified_count || 0 },
          { label: "No catalogadas", value: latest.unknown_cows_estimated || 0 },
          { label: "Diferencia", value: Math.abs(latest.count_delta || 0) },
        ])}
      </article>
    </section>
  `;
}

function renderCatalog() {
  const cards = dashboardData.catalog.map((cow) => `
    <article class="cow-card">
      <img src="${cow.cover_url}" alt="${cow.name}">
      <div>
        <span class="status-ok">${cow.status}</span>
        <h3>${cow.name}</h3>
        <p>${cow.phenotype}</p>
        <small>${cow.photo_count} foto(s) · embeddings ${cow.embedding_status}</small>
        <button class="secondary" onclick="deleteCow('${cow.name}')">Eliminar identidad</button>
      </div>
    </article>
  `).join("");
  $("#view-catalogo").innerHTML = `
    <section class="panel">
      <span class="eyebrow">Galería del usuario</span>
      <h2>Catálogo de vacas</h2>
      <p>Estas identidades se usan como galería de referencia para reidentificación. Para cada vaca se guardan fotos y estado de embeddings.</p>
    </section>
    <div class="cow-grid">${cards || "<p>No hay vacas catalogadas.</p>"}</div>
    <section class="panel">
      <h2>Agregar identidad</h2>
      <form id="cowForm" class="form-grid">
        <input name="cow_name" placeholder="Nombre de la vaca" required>
        <input name="phenotype" placeholder="Descripción visual">
        <label class="file-control full"><span>Elegir fotos de referencia</span><input name="cow_photos" type="file" accept="image/*" multiple required></label>
        <button class="primary full" type="submit">Guardar vaca catalogada</button>
      </form>
    </section>
  `;
  $("#cowForm").addEventListener("submit", saveCow);
  bindFileControls();
}

function renderProcess() {
  const state = dashboardData.state || {};
  const running = isRunningState(state);
  const progress = running ? state.progress || 1 : 0;
  $("#view-conteo").innerHTML = `
    <section class="process-shell">
      <div class="panel process-card">
        <span class="eyebrow">Procesamiento</span>
        <h2>Conteo CowTrack</h2>
        <p>Cargá el video del rodeo, indicá la referencia esperada y ejecutá el análisis. Si hay un proceso en curso, el sistema bloquea nuevas cargas hasta finalizar o reiniciar.</p>
        <form id="runForm" class="process-form">
          <label class="file-control primary-upload"><span>Elegir video del rodeo</span><input name="video_file" type="file" accept="video/*" required></label>
          <input name="expected_total_cows" type="number" min="1" placeholder="Cantidad total del rodeo (ej. 13)" required>
          <input name="output_name" placeholder="Nombre del informe (ej. Reidentificación ${todaySlug()})">
          <button class="primary" type="submit" ${running ? "disabled" : ""}>${running ? "Procesando..." : "Reidentificar rodeo"}</button>
        </form>
      </div>
      <div class="panel progress-card">
        <div class="progress-head">
          <h2>Estado del análisis</h2>
          <strong>${progress}%</strong>
        </div>
        <div class="formal-progress"><i style="width:${progress}%"></i></div>
        <p id="friendlyStep">${running ? state.step || "Procesando video" : "Listo para iniciar"}</p>
        ${running ? `<button class="secondary" onclick="resetAnalysis()">Cancelar procesamiento</button>` : ""}
      </div>
    </section>
  `;
  $("#runForm").addEventListener("submit", startRun);
  bindFileControls();
}

function cowFaces(names = []) {
  const selected = dashboardData.catalog.filter((cow) => names.includes(cow.name));
  if (!selected.length) return "";
  return `<div class="cow-grid compact-cows">${selected.map((cow) => `
    <article class="cow-card"><img src="${cow.cover_url}" alt="${cow.name}"><div><h3>${cow.name}</h3><span class="status-ok">reidentificada</span></div></article>
  `).join("")}</div>`;
}

function renderReports() {
  const reports = dashboardData.reports.map((r) => {
    const contact = fileLink(r.contact_sheet_path);
    const video = fileLink(r.video_path);
    const showContact = r.artifact_status?.contact_sheet && Number(r.estimated_total_cows || 0) > 0;
    return `
      <article class="report-card ${r.status_tone === "warn" ? "report-warn" : ""} ${r.status_tone === "bad" ? "report-bad" : ""}">
        <header class="report-header">
          <div>
            <span class="eyebrow">Informe CowTrack</span>
            <h3>${r.title}</h3>
            <p>${r.date}</p>
          </div>
          <strong>${r.status_label || "Informe"}</strong>
        </header>
        <div class="report-body">
          <section class="report-summary">
            ${metric("Conteo obtenido", `${r.estimated_total_cows ?? "-"} vacas`, `Referencia: ${r.expected_total_cows ?? "-"} vacas`)}
            ${metric("Diferencia", r.count_delta ?? "-", "obtenido vs referencia")}
            ${metric("Confiabilidad", `${r.count_accuracy_percent ?? "-"}%`, r.status_label || "")}
            ${metric("Frames", r.technical?.processed_frames ?? "-", "procesados")}
          </section>
          <section class="report-two-col">
            <div>
              <h4>Reidentificación</h4>
              ${cowFaces(r.reidentified_cows)}
              <p>Catalogadas localizadas: ${(r.reidentified_cows || []).join(", ") || "ninguna"}. No catalogadas estimadas: ${r.unknown_cows_estimated ?? "-"}.</p>
            </div>
            <div>
              <h4>Lectura técnica</h4>
              ${barChart([
                { label: "Margarita", value: Math.round((r.technical?.known_hit_ratio?.Margarita || 0) * 100) },
                { label: "Maria", value: Math.round((r.technical?.known_hit_ratio?.Maria || 0) * 100) },
                { label: "Marta", value: Math.round((r.technical?.known_hit_ratio?.Marta || 0) * 100) },
              ])}
            </div>
          </section>
          ${showContact && contact ? `<section><h4>Evidencia visual</h4><div class="friendly-report"><img src="${contact}" alt="Capturas del análisis"></div></section>` : ""}
        </div>
        <div class="report-actions">
          <a class="primary report-link" href="/api/report_pdf?id=${r.id}" target="_blank">Ver informe PDF</a>
          ${video && r.artifact_status?.video ? `<a class="primary report-link" href="${video}" target="_blank" download>Descargar video</a>` : ""}
          <button class="primary telegram-button" onclick="openTelegram()"><span class="plane-icon"></span>Enviar por Telegram</button>
        </div>
      </article>
    `;
  }).join("");
  $("#view-reportes").innerHTML = `
    <section class="panel">
      <span class="eyebrow">Historial</span>
      <h2>Reportes ejecutivos</h2>
      <p>Cada informe resume conteo, reidentificación, frames procesados, estado de tracking y evidencia visual. Los reportes quedan conservados para comparar resultados y auditar el historial.</p>
    </section>
    <div class="report-grid">${reports || "<p>Aún no hay reportes.</p>"}</div>
  `;
}

function renderMetrics() {
  const latest = latestValidReport();
  const tech = latest.technical || {};
  const visible = tech.per_frame_detection_summary || {};
  const modes = {
    general: {
      title: "Lectura general del rodeo",
      text: "Resume el resultado operativo que necesita el productor: conteo, diferencia contra referencia y vacas no catalogadas.",
      donuts: [
        [reportAccuracy(latest), "confiabilidad de conteo"],
        [latest.count_delta === 0 ? 100 : Math.max(0, 100 - Math.abs(latest.count_delta || 0) * 10), "ajuste a referencia"],
      ],
      bars: [
        { label: "Conteo", value: latest.estimated_total_cows || 0 },
        { label: "Referencia", value: latest.expected_total_cows || 0 },
        { label: "No catalogadas", value: latest.unknown_cows_estimated || 0 },
      ],
    },
    reid: {
      title: "Reidentificación individual",
      text: "Muestra si las vacas catalogadas fueron localizadas y con qué continuidad aparecen en el análisis.",
      donuts: [
        [((latest.reidentified_count || 0) / Math.max(1, dashboardData.catalog.length)) * 100, "catálogo localizado"],
        [tech.tracking_ok ? 100 : 0, "seguimiento estable"],
      ],
      bars: Object.entries(tech.known_hit_ratio || {}).map(([label, value]) => ({ label, value: Math.round(Number(value || 0) * 100) })),
    },
    deteccion: {
      title: "Detecciones por frame",
      text: "Describe la lectura visual por cuadro. Sirve para entender estabilidad, oclusiones y variaciones de cámara.",
      donuts: [
        [Math.min(100, Number(visible.mean_visible_detections || 0) * 8), "densidad promedio"],
        [Math.min(100, Number(visible.p95_visible_detections || 0) * 7), "p95 visible"],
      ],
      bars: [
        { label: "Promedio", value: Number(visible.mean_visible_detections || 0).toFixed(1) },
        { label: "Mediana", value: Number(visible.median_visible_detections || 0).toFixed(1) },
        { label: "P95", value: Number(visible.p95_visible_detections || 0).toFixed(1) },
        { label: "Máximo", value: visible.max_visible_detections || 0 },
      ],
    },
    tecnica: {
      title: "Calidad técnica del análisis",
      text: "Agrupa señales internas del procesamiento: frames, scores, tracking y consistencia de las identidades.",
      donuts: [
        [tech.ready_for_render ? 100 : 70, "listo para informe"],
        [tech.count_within_tolerance ? 100 : reportAccuracy(latest), "conteo dentro de rango"],
      ],
      bars: Object.entries(tech.identity_scores || {}).map(([label, value]) => ({ label, value: Math.round(Number(value || 0) * 100) })),
    },
  };
  const selected = modes[metricMode] || modes.general;
  $("#view-metricas").innerHTML = `
    <section class="panel metrics-hero">
      <div>
        <span class="eyebrow">Métricas</span>
        <h2>Centro de análisis CowTrack</h2>
        <p>Explorá el último informe desde distintas lecturas: conteo general, vacas reidentificadas, detecciones visibles y señales técnicas del procesamiento.</p>
      </div>
      <div class="metric-tabs">
        ${Object.keys(modes).map((key) => `<button class="${metricMode === key ? "active" : ""}" data-metric-mode="${key}">${key === "reid" ? "Reidentificación" : key === "deteccion" ? "Detección" : key === "tecnica" ? "Técnica" : "General"}</button>`).join("")}
      </div>
    </section>
    <section class="metrics-layout interactive-metrics">
      <article class="panel">
        <span class="eyebrow">Vista seleccionada</span>
        <h2>${selected.title}</h2>
        <p>${selected.text}</p>
        <div class="mini-metrics">
          ${selected.donuts.map(([value, label]) => donut(value, label)).join("")}
        </div>
      </article>
      <article class="panel">
        <h2>Distribución</h2>
        ${barChart(selected.bars.length ? selected.bars : [{ label: "Sin datos", value: 0 }])}
      </article>
      <article class="panel wide metric-detail-grid">
        ${metric("Frames procesados", tech.processed_frames ?? "-", "último informe")}
        ${metric("Conteo obtenido", `${latest.estimated_total_cows ?? "-"} vacas`, `Referencia: ${latest.expected_total_cows ?? "-"} vacas`)}
        ${metric("Reidentificadas", latest.reidentified_count ?? 0, (latest.reidentified_cows || []).join(", ") || "Sin localizaciones")}
        ${metric("Estado", latest.status_label || "-", "validación operativa")}
        <p class="technical-note full">${tech.metric_note || "Las métricas se calculan a partir del informe técnico generado por el procesamiento."}</p>
      </article>
    </section>
  `;
  $$("[data-metric-mode]").forEach((btn) => btn.addEventListener("click", () => {
    metricMode = btn.dataset.metricMode;
    renderMetrics();
  }));
}

function renderSupport() {
  $("#view-soporte").innerHTML = `
    <section class="panel">
      <span class="eyebrow">Soporte técnico</span>
      <h2>Centro de ayuda CowTrack</h2>
      <p>Registrá una consulta asociada a tu cuenta. El equipo de soporte podrá revisar el caso, el último reporte y el contexto de procesamiento.</p>
      <form id="supportForm" class="form-grid">
        <input name="asunto" placeholder="Asunto" required>
        <select name="tipo_consulta">
          <option>Consulta sobre procesamiento</option>
          <option>Problema con un reporte</option>
          <option>Alta de nuevas vacas catalogadas</option>
          <option>Soporte comercial</option>
        </select>
        <textarea class="full" name="detalle" placeholder="Describí el problema o consulta"></textarea>
        <button class="primary full" type="submit">Enviar solicitud</button>
      </form>
    </section>
  `;
  $("#supportForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    showMessage("Solicitud registrada", "El pedido de soporte quedó registrado y asociado a tu usuario.");
    event.currentTarget.reset();
  });
}

function renderFaq() {
  $("#view-faq").innerHTML = `
    <section class="panel">
      <span class="eyebrow">FAQ</span>
      <h2>Preguntas frecuentes</h2>
      <div class="faq-list">
        <article><h3>¿Qué necesito para usar CowTrack?</h3><p>Un video del rodeo capturado con dron y, si se desea reidentificación, una galería de fotos por vaca catalogada.</p></article>
        <article><h3>¿Por qué se pide una cantidad esperada?</h3><p>La referencia permite contrastar el conteo automático contra el conocimiento del productor y estimar si el resultado operativo es aceptable.</p></article>
        <article><h3>¿Qué pasa si una vaca sale del plano?</h3><p>El tracking intenta sostener la identidad mientras hay evidencia visual suficiente. Si el animal sale de cámara, la continuidad se audita en el reporte.</p></article>
        <article><h3>¿CowTrack funciona sin internet?</h3><p>La plataforma está pensada para escenarios online y offline. Puede operar localmente en zonas rurales y sincronizar reportes cuando hay conectividad.</p></article>
      </div>
    </section>
  `;
}

function renderAll() {
  renderOverview();
  renderCatalog();
  renderProcess();
  renderReports();
  renderMetrics();
  renderSupport();
  renderFaq();
}

async function loadDashboard() {
  dashboardData = await api("/api/dashboard");
  currentUser = currentUser || "admin";
  $("#helloTitle").textContent = `¡Hola, ${currentUser}!`;
  renderAll();
  updatePersistentProgress(dashboardData.state);
  ensureStatusPolling();
}

async function saveCow(event) {
  event.preventDefault();
  await api("/api/catalog", { method: "POST", body: formData(event.currentTarget) });
  showMessage("Vaca catalogada", "La identidad quedó guardada y disponible para próximos conteos.");
  await loadDashboard();
  setView("catalogo");
}

async function deleteCow(name) {
  const body = new URLSearchParams({ cow_name: name });
  await api("/api/catalog/delete", { method: "POST", body });
  await loadDashboard();
  setView("catalogo");
}

async function resetAnalysis() {
  if (statusTimer) clearInterval(statusTimer);
  await api("/api/reset", { method: "POST" });
  dashboardData.state = {};
  await loadDashboard();
  setView("conteo");
}

async function startRun(event) {
  event.preventDefault();
  const form = event.currentTarget;
  await api("/api/run", { method: "POST", body: formData(form) });
  dashboardData.state = { status: "queued", progress: 1, step: "Preparando el análisis" };
  renderProcess();
  setView("conteo");
  if (statusTimer) clearInterval(statusTimer);
  statusTimer = setInterval(updateStatus, 1500);
  updateStatus();
}

async function updateStatus() {
  const state = await api("/api/status");
  dashboardData.state = state;
  updatePersistentProgress(state);
  const progress = isRunningState(state) ? state.progress || 1 : 0;
  $(".formal-progress i")?.style.setProperty("width", `${progress}%`);
  const progressStrong = $(".progress-head strong");
  if (progressStrong) progressStrong.textContent = `${progress}%`;
  if ($("#friendlyStep")) $("#friendlyStep").textContent = state.step || "Procesando";
  if (state.status === "completed" || state.status === "failed") {
    clearInterval(statusTimer);
    await loadDashboard();
    if (state.status === "completed") {
      const latest = dashboardData.reports[0] || {};
      const title = "Reporte listo";
      const text = latest.status_tone === "bad"
        ? "El procesamiento terminó, pero no se detectaron vacas. Revisá el video cargado antes de usar el informe."
        : "El procesamiento terminó. Revisá el informe ejecutivo, el video renderizado y las métricas del análisis.";
      showMessage(title, text);
      await api("/api/reset", { method: "POST" });
      dashboardData.state = {};
      setView("reportes");
    } else {
      showMessage("No se pudo completar el análisis", state.error || "Revisá el video seleccionado y volvé a intentarlo.", "CowTrack");
      await api("/api/reset", { method: "POST" });
      dashboardData.state = {};
      renderProcess();
      setView("conteo");
    }
  }
}

async function sendTelegram(event) {
  event.preventDefault();
  const body = new URLSearchParams(new FormData(event.currentTarget));
  const result = await api("/api/telegram", { method: "POST", body });
  closeTelegram();
  showMessage("Telegram actualizado", result.message || "El reporte fue enviado correctamente.");
}

function bindFileControls() {
  $$(".file-control input[type='file']").forEach((input) => {
    input.addEventListener("change", () => {
      const count = input.files ? input.files.length : 0;
      const label = input.closest(".file-control")?.querySelector("span");
      if (label) label.textContent = count === 1 ? input.files[0].name : `${count} archivos seleccionados`;
    });
  });
}

async function boot() {
  await api("/api/reset", { method: "POST" }).catch(() => {});
  $$(".close-offer").forEach((btn) => btn.addEventListener("click", () => $("#launchOffer").classList.add("hidden")));
  $("#introClose").addEventListener("click", () => { $("#introVideo").classList.add("hidden"); $("#adVideo").pause(); showOfferOnce(); });
  $("#adVideo").addEventListener("ended", () => { $("#introVideo").classList.add("hidden"); showOfferOnce(); });
  $$("[data-go]").forEach((el) => el.addEventListener("click", (event) => {
    event.preventDefault();
    showPublicPage(el.dataset.go);
    window.location.hash = el.dataset.go;
  }));
  window.addEventListener("hashchange", openHashPage);
  $$("[data-protected]").forEach((el) => el.addEventListener("click", () => currentUser ? showDashboard(el.dataset.protected) : openLogin()));
  $("#loginOpen").addEventListener("click", openLogin);
  $("#loginClose").addEventListener("click", closeLogin);
  $("#registerOpen").addEventListener("click", openRegister);
  $("#registerClose").addEventListener("click", closeRegister);
  $("#dashboardOpen").addEventListener("click", async () => { await loadDashboard(); showDashboard(); });
  $("#messageClose").addEventListener("click", hideMessage);
  $("#messageOk").addEventListener("click", hideMessage);
  $("#telegramClose").addEventListener("click", closeTelegram);
  $("#telegramForm").addEventListener("submit", sendTelegram);
  $$("[data-social]").forEach((btn) => btn.addEventListener("click", async () => {
    const body = new URLSearchParams({ provider: btn.dataset.social });
    const result = await api("/api/social_login", { method: "POST", body });
    currentUser = result.user;
    closeRegister();
    await loadDashboard();
    showDashboard();
  }));
  $("#logoutButton").addEventListener("click", async () => {
    await api("/api/logout", { method: "POST" });
    currentUser = null;
    showPublicPage("home");
  });
  $("#resetButton").addEventListener("click", resetAnalysis);
  $$(".dash-sidebar nav button").forEach((btn) => btn.addEventListener("click", () => setView(btn.dataset.view)));
  $("#loginForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const result = await api("/api/login", { method: "POST", body: formData(event.currentTarget) });
    currentUser = result.user || "admin";
    closeLogin();
    await loadDashboard();
    showDashboard();
  });
  $("#registerForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const result = await api("/api/register", { method: "POST", body: formData(event.currentTarget) });
    currentUser = result.user;
    closeRegister();
    await loadDashboard();
    showDashboard();
  });
  $("#contactForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = await api("/api/contact", { method: "POST", body: formData(event.currentTarget) });
    showMessage("Consulta recibida", data.message);
    event.currentTarget.reset();
  });
  const session = await api("/api/session");
  if (session.authenticated) {
    currentUser = session.user || "admin";
    await loadDashboard();
    const hashView = (window.location.hash || "").replace("#dashboard-", "");
    const view = document.getElementById(`view-${hashView}`) ? hashView : "overview";
    showDashboard(view);
  } else {
    showPublicPage((window.location.hash || "#home").replace("#", ""));
    maybeShowIntro();
  }
}

boot().catch((error) => showMessage("Ocurrió un problema", error.message));
