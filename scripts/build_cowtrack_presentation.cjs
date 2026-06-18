const fs = require("fs");
const path = require("path");
const PptxGenJS = require("/Users/emilianoorlando/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const repoRoot = path.resolve(__dirname, "..");
const outPath = path.join(repoRoot, "docs", "CowTrack_MVP_presentacion.pptx");
const finalPreview = path.join(repoRoot, "reports", "final", "16_erondina_final_render_preview.jpg");
const contactSheet = path.join(repoRoot, "reports", "final", "16_erondina_final_contact_sheet.jpg");
const candidateSheet = path.join(repoRoot, "reports", "final", "16_erondina_final_candidate_sheet.jpg");
const coverImage = path.join(repoRoot, "screenshots", "Captura de pantalla 2026-06-02 a las 11.19.54 p. m..png");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "CowTrack MVP";
pptx.company = "Instituto de Formación Técnica Superior Número 11";
pptx.subject = "Detección, tracking y reidentificación de vacas en video aéreo";
pptx.title = "CowTrack MVP";
pptx.lang = "es-AR";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "es-AR",
};

const SLIDE_W = 13.333;
const SLIDE_H = 7.5;
const C = {
  cream: "F6F1E7",
  paper: "FFF9EF",
  dark: "172719",
  green: "23452C",
  green2: "53733A",
  grass: "78904B",
  clay: "B06B3F",
  brown: "7A4A2D",
  straw: "D9B85F",
  yellow: "F8E94B",
  blue: "2F86C6",
  black: "0B0B0A",
  white: "FFFFFF",
  muted: "6F7468",
  red: "B75C3A",
};

function slideBase(slide, title, kicker = "CowTrack MVP") {
  slide.background = { color: C.cream };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: SLIDE_W,
    h: 0.18,
    fill: { color: C.green },
    line: { color: C.green },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 7.18,
    w: SLIDE_W,
    h: 0.32,
    fill: { color: C.green, transparency: 8 },
    line: { color: C.green },
  });
  slide.addText(kicker.toUpperCase(), {
    x: 0.55,
    y: 0.42,
    w: 3.3,
    h: 0.22,
    fontFace: "Aptos",
    fontSize: 8,
    color: C.grass,
    bold: true,
    charSpace: 1.2,
    margin: 0,
  });
  slide.addText(title, {
    x: 0.55,
    y: 0.68,
    w: 8.5,
    h: 0.55,
    fontFace: "Aptos Display",
    fontSize: 27,
    bold: true,
    color: C.dark,
    margin: 0,
    fit: "shrink",
  });
  slide.addText("IFTS N° 11 · Ciencia de Datos e IA · Junio 2026", {
    x: 0.55,
    y: 7.28,
    w: 8,
    h: 0.15,
    fontSize: 7.5,
    color: "E7EAD8",
    margin: 0,
  });
}

function addPill(slide, text, x, y, w, fill, color = C.white) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h: 0.34,
    rectRadius: 0.05,
    fill: { color: fill },
    line: { color: fill },
  });
  slide.addText(text, {
    x: x + 0.12,
    y: y + 0.08,
    w: w - 0.24,
    h: 0.14,
    fontSize: 8.5,
    bold: true,
    color,
    margin: 0,
    align: "center",
    fit: "shrink",
  });
}

function addMetric(slide, value, label, x, y, w, accent, note = "") {
  slide.addShape(pptx.ShapeType.rect, {
    x,
    y,
    w,
    h: 1.28,
    fill: { color: C.paper },
    line: { color: "E7D7B9", width: 1 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x,
    y,
    w: 0.08,
    h: 1.28,
    fill: { color: accent },
    line: { color: accent },
  });
  slide.addText(value, {
    x: x + 0.22,
    y: y + 0.17,
    w: w - 0.35,
    h: 0.4,
    fontFace: "Aptos Display",
    fontSize: 25,
    bold: true,
    color: C.dark,
    margin: 0,
    fit: "shrink",
  });
  slide.addText(label, {
    x: x + 0.22,
    y: y + 0.67,
    w: w - 0.35,
    h: 0.22,
    fontSize: 9.5,
    bold: true,
    color: C.green,
    margin: 0,
    fit: "shrink",
  });
  if (note) {
    slide.addText(note, {
      x: x + 0.22,
      y: y + 0.96,
      w: w - 0.35,
      h: 0.18,
      fontSize: 7.4,
      color: C.muted,
      margin: 0,
      fit: "shrink",
    });
  }
}

function addBody(slide, lines, x, y, w, h, opts = {}) {
  slide.addText(lines.join("\n"), {
    x,
    y,
    w,
    h,
    fontSize: opts.fontSize || 15,
    color: opts.color || C.dark,
    breakLine: false,
    fit: "shrink",
    margin: 0,
    paraSpaceAfterPt: opts.paraSpaceAfterPt || 8,
    bullet: opts.bullet || undefined,
  });
}

function addSectionTag(slide, text, x, y, fill = C.green) {
  slide.addShape(pptx.ShapeType.rect, {
    x,
    y,
    w: 0.08,
    h: 0.34,
    fill: { color: fill },
    line: { color: fill },
  });
  slide.addText(text.toUpperCase(), {
    x: x + 0.16,
    y: y + 0.06,
    w: 2.5,
    h: 0.14,
    fontSize: 8,
    bold: true,
    color: fill,
    charSpace: 1,
    margin: 0,
  });
}

function addArrow(slide, x1, y1, x2, y2, color = C.straw) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1,
    y: y1,
    w: x2 - x1,
    h: y2 - y1,
    line: { color, width: 2, beginArrowType: "none", endArrowType: "triangle" },
  });
}

function addNode(slide, text, x, y, w, h, fill, color = C.white) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.05,
    fill: { color: fill },
    line: { color: fill },
  });
  slide.addText(text, {
    x: x + 0.12,
    y: y + 0.12,
    w: w - 0.24,
    h: h - 0.18,
    fontSize: 10.5,
    bold: true,
    color,
    align: "center",
    valign: "mid",
    margin: 0,
    fit: "shrink",
  });
}

function addSoftBand(slide, y, color = C.straw) {
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y,
    w: SLIDE_W,
    h: 0.06,
    fill: { color, transparency: 20 },
    line: { color, transparency: 100 },
  });
}

// 1. Cover
{
  const slide = pptx.addSlide();
  slide.background = { color: C.dark };
  if (fs.existsSync(coverImage)) {
    slide.addImage({ path: coverImage, x: 0, y: 0, w: SLIDE_W, h: SLIDE_H });
  } else {
    slide.addImage({ path: finalPreview, x: 0, y: 0, w: SLIDE_W, h: SLIDE_H });
  }
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: SLIDE_W,
    h: SLIDE_H,
    fill: { color: C.dark, transparency: 16 },
    line: { color: C.dark, transparency: 100 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: SLIDE_W,
    h: 0.18,
    fill: { color: C.yellow },
    line: { color: C.yellow },
  });
  addPill(slide, "MVP COMPLETO", 0.68, 0.72, 1.45, C.yellow, C.black);
  slide.addText("CowTrack MVP", {
    x: 0.68,
    y: 1.28,
    w: 7.1,
    h: 0.9,
    fontFace: "Aptos Display",
    fontSize: 48,
    bold: true,
    color: C.white,
    margin: 0,
  });
  slide.addText("Detección, conteo, tracking y reidentificación de vacas en video aéreo", {
    x: 0.72,
    y: 2.15,
    w: 7.8,
    h: 0.56,
    fontSize: 20,
    color: "F8F1D8",
    margin: 0,
    fit: "shrink",
  });
  slide.addText("Emiliano Orlando · Pablo Canteros · Ramiro Ottone Villar · Verónica Arce", {
    x: 0.72,
    y: 6.18,
    w: 7.6,
    h: 0.26,
    fontSize: 11.5,
    color: C.white,
    bold: true,
    margin: 0,
  });
  slide.addText("Instituto de Formación Técnica Superior Número 11 · Ciencia de Datos e IA", {
    x: 0.72,
    y: 6.52,
    w: 7.4,
    h: 0.2,
    fontSize: 9.5,
    color: "F0E4BC",
    margin: 0,
  });
  addMetric(slide, "100%", "Re-ID", 9.25, 4.55, 1.5, C.yellow, "3/3 identidades");
  addMetric(slide, "0", "ID switches", 10.95, 4.55, 1.5, C.blue, "Marta, Maria y Margarita");
}

// 2. Problem
{
  const slide = pptx.addSlide();
  slideBase(slide, "Problema y oportunidad", "Contexto agropecuario");
  addSoftBand(slide, 1.36);
  addBody(
    slide,
    [
      "En ganadería, el seguimiento individual suele depender de observación manual, recorridas o dispositivos físicos.",
      "El video aéreo con dron permite observar el rodeo, pero introduce desafíos: cambios de escala, giros de cámara, oclusiones y animales similares.",
      "CowTrack transforma ese registro visual en una salida auditable: detección, tracking, conteo operativo y reidentificación individual.",
    ],
    0.75,
    1.78,
    6.05,
    3.0,
    { fontSize: 16, paraSpaceAfterPt: 11 }
  );
  slide.addShape(pptx.ShapeType.rect, {
    x: 7.35,
    y: 1.72,
    w: 4.9,
    h: 3.75,
    fill: { color: C.paper },
    line: { color: "E5D4AD", width: 1 },
  });
  addSectionTag(slide, "Desafíos del campo", 7.72, 2.03, C.clay);
  addBody(
    slide,
    [
      "• Vacas parcialmente tapadas",
      "• Movimiento del dron",
      "• Animales que entran o salen de plano",
      "• Diferencia entre contar e identificar",
      "• IDs internos fragmentados por tracking",
    ],
    7.72,
    2.55,
    4.1,
    2.4,
    { fontSize: 15 }
  );
}

// 3. Objectives
{
  const slide = pptx.addSlide();
  slideBase(slide, "Objetivo del MVP", "Alcance funcional");
  slide.addText("Resultado esperado", {
    x: 0.8,
    y: 1.58,
    w: 4.4,
    h: 0.36,
    fontSize: 20,
    bold: true,
    color: C.green,
    margin: 0,
  });
  addBody(
    slide,
    [
      "Procesar video aéreo del campo Erondina y producir un render HD donde:",
      "• cada vaca detectada tenga bounding box;",
      "• Marta, Maria y Margarita estén identificadas con etiqueta priorizada;",
      "• las identidades catalogadas se mantengan estables en el tiempo;",
      "• el conteo general quede medido y documentado.",
    ],
    0.8,
    2.05,
    5.7,
    3.0,
    { fontSize: 15.5, paraSpaceAfterPt: 7 }
  );
  addMetric(slide, "Marta", "identidad catalogada", 7.0, 1.55, 2.1, C.blue, "castaña");
  addMetric(slide, "Maria", "identidad catalogada", 9.35, 1.55, 2.1, C.yellow, "negra");
  addMetric(slide, "Margarita", "identidad catalogada", 7.0, 3.28, 2.1, "31D75C", "castaña");
  addMetric(slide, "13 vs 21", "conteo documentado", 9.35, 3.28, 2.1, C.clay, "real vs etiquetas");
}

// 4. Data and models
{
  const slide = pptx.addSlide();
  slideBase(slide, "Datos, modelos y galería", "Base técnica");
  slide.addText("La solución combina un extractor Re-ID general con una galería específica del campo Erondina.", {
    x: 0.72,
    y: 1.42,
    w: 9.3,
    h: 0.32,
    fontSize: 16,
    color: C.muted,
    margin: 0,
  });
  addNode(slide, "OpenCows\nDataset general", 0.82, 2.12, 2.05, 1.0, C.green);
  addArrow(slide, 2.92, 2.62, 3.58, 2.62);
  addNode(slide, "Entrenamiento\nRe-ID", 3.68, 2.12, 1.9, 1.0, C.brown);
  addArrow(slide, 5.62, 2.62, 6.28, 2.62);
  addNode(slide, "mi_modelo_reid.pt", 6.38, 2.12, 2.05, 1.0, C.clay);
  addArrow(slide, 8.48, 2.62, 9.14, 2.62);
  addNode(slide, "Embeddings\nbase", 9.24, 2.12, 1.75, 1.0, C.grass);
  addNode(slide, "Fotos extraídas\ndel campo Erondina", 1.2, 4.32, 2.45, 1.0, C.straw, C.black);
  addArrow(slide, 3.72, 4.82, 4.38, 4.82);
  addNode(slide, "Recorte y filtrado\nde referencias", 4.48, 4.32, 2.25, 1.0, C.green2);
  addArrow(slide, 6.82, 4.82, 7.48, 4.82);
  addNode(slide, "Galería Erondina\nMarta · Maria · Margarita", 7.58, 4.32, 3.05, 1.0, C.yellow, C.black);
  slide.addText("Decisión clave: aislar referencias visuales enfocadas para que la comparación de embeddings sea más robusta.", {
    x: 0.95,
    y: 6.08,
    w: 10.7,
    h: 0.28,
    fontSize: 13,
    bold: true,
    color: C.green,
    margin: 0,
    fit: "shrink",
  });
}

// 5. Pipeline architecture
{
  const slide = pptx.addSlide();
  slideBase(slide, "Arquitectura del pipeline final", "De video a render HD");
  const y1 = 1.65;
  const nodes = [
    ["Video\nErondina", 0.65, y1, C.green],
    ["YOLOv8\nDetección", 2.35, y1, C.brown],
    ["Tracking\nbase", 4.05, y1, C.clay],
    ["Recortes\npor vaca", 5.75, y1, C.grass],
    ["Embeddings\ncuerpo + cabeza", 7.45, y1, C.green2],
    ["FAISS /\ncoseno", 9.15, y1, C.straw],
    ["IDs globales\nbloqueados", 10.85, y1, C.blue],
  ];
  nodes.forEach(([t, x, y, fill]) => addNode(slide, t, x, y, 1.35, 0.86, fill, fill === C.straw ? C.black : C.white));
  for (let i = 0; i < nodes.length - 1; i += 1) addArrow(slide, nodes[i][1] + 1.38, y1 + 0.43, nodes[i + 1][1] - 0.06, y1 + 0.43, C.straw);
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.75,
    y: 3.35,
    w: 11.7,
    h: 2.2,
    fill: { color: C.paper },
    line: { color: "E2D0A8", width: 1 },
  });
  addSectionTag(slide, "Lógica central", 1.08, 3.72, C.green);
  addBody(
    slide,
    [
      "Una vez reconocida la identidad, el render final no decide desde cero en cada frame:",
      "usa la identidad de la vaca reconocida y la sigue temporalmente.",
      "Esto reduce cambios de ID y prioriza las etiquetas de Marta, Maria y Margarita por encima de vacas no catalogadas.",
    ],
    1.08,
    4.18,
    10.8,
    1.0,
    { fontSize: 15, paraSpaceAfterPt: 6 }
  );
}

// 6. Re-ID validation
{
  const slide = pptx.addSlide();
  slideBase(slide, "Validación Re-ID: tres identidades", "Marta · Maria · Margarita");
  slide.addImage({ path: contactSheet, x: 0.72, y: 1.45, w: 5.6, h: 4.3 });
  slide.addShape(pptx.ShapeType.rect, {
    x: 6.75,
    y: 1.52,
    w: 5.55,
    h: 4.15,
    fill: { color: C.paper },
    line: { color: "E7D7B9", width: 1 },
  });
  addSectionTag(slide, "Scores finales", 7.12, 1.85, C.green);
  const rows = [
    ["Margarita", "0.9269", "AP@0.5 = 1.0", "31D75C"],
    ["Maria", "0.8973", "AP@0.5 = 1.0", C.yellow],
    ["Marta", "0.9459", "AP@0.5 = 1.0", C.blue],
  ];
  rows.forEach((r, i) => {
    const y = 2.35 + i * 0.85;
    slide.addShape(pptx.ShapeType.rect, { x: 7.12, y, w: 0.13, h: 0.46, fill: { color: r[3] }, line: { color: r[3] } });
    slide.addText(r[0], { x: 7.36, y: y + 0.03, w: 1.6, h: 0.22, fontSize: 13, bold: true, color: C.dark, margin: 0 });
    slide.addText(r[1], { x: 9.0, y: y + 0.03, w: 0.9, h: 0.22, fontSize: 13, bold: true, color: C.green, margin: 0 });
    slide.addText(r[2], { x: 10.1, y: y + 0.05, w: 1.5, h: 0.18, fontSize: 9.5, color: C.muted, margin: 0 });
  });
  addMetric(slide, "100%", "Precision Re-ID", 7.12, 4.9, 1.62, C.green, "TP/(TP+FP)");
  addMetric(slide, "100%", "Recall Re-ID", 8.94, 4.9, 1.62, C.green, "TP/(TP+FN)");
  addMetric(slide, "100%", "mAP@0.5 Re-ID", 10.76, 4.9, 1.62, C.green, "3 identidades");
}

// 7. Render evidence
{
  const slide = pptx.addSlide();
  slideBase(slide, "Resultado visual del render HD", "Evidencia de presentación");
  slide.addImage({ path: finalPreview, x: 0.52, y: 1.38, w: 12.28, h: 4.6 });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.52,
    y: 6.12,
    w: 12.28,
    h: 0.58,
    fill: { color: C.green },
    line: { color: C.green },
  });
  slide.addText("Etiquetas priorizadas para vacas reidentificadas + bounding boxes para vacas no catalogadas", {
    x: 0.82,
    y: 6.32,
    w: 10.9,
    h: 0.16,
    fontSize: 12,
    bold: true,
    color: C.white,
    margin: 0,
    fit: "shrink",
  });
}

// 8. Metrics
{
  const slide = pptx.addSlide();
  slideBase(slide, "Métricas del MVP final", "Validación cuantitativa");
  addMetric(slide, "47.01 s", "Duración del video final", 0.72, 1.5, 2.45, C.straw, "1409 frames · 29.97 FPS");
  addMetric(slide, "1920×1080", "Resolución HD", 3.45, 1.5, 2.45, C.green, "render final");
  addMetric(slide, "99.81%", "Presencia promedio", 6.18, 1.5, 2.45, C.blue, "vacas catalogadas");
  addMetric(slide, "0", "ID switches Re-ID", 8.91, 1.5, 2.45, C.green, "identidades bloqueadas");
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.72,
    y: 3.55,
    w: 11.62,
    h: 2.65,
    fill: { color: C.paper },
    line: { color: "E1D0AB", width: 1 },
  });
  slide.addText("Tabla solicitada de validación Re-ID", { x: 1.05, y: 3.85, w: 4.2, h: 0.22, fontSize: 15, bold: true, color: C.green, margin: 0 });
  const headers = ["Métrica evaluada", "Valor obtenido", "Umbral", "Estado"];
  const vals = [
    ["Precisión", "100.0%", "> 80.0%", "Superado"],
    ["Exhaustividad", "100.0%", "> 80.0%", "Superado"],
    ["mAP@0.5 Re-ID", "100.0%", "> 85.0%", "Superado"],
  ];
  const colX = [1.05, 4.0, 6.65, 8.75];
  headers.forEach((h, i) => slide.addText(h, { x: colX[i], y: 4.42, w: i === 0 ? 2.4 : 1.8, h: 0.16, fontSize: 8.5, bold: true, color: C.muted, margin: 0 }));
  vals.forEach((row, r) => {
    const y = 4.84 + r * 0.43;
    row.forEach((v, i) => slide.addText(v, { x: colX[i], y, w: i === 0 ? 2.4 : 1.8, h: 0.16, fontSize: 11, bold: i === 1 || i === 3, color: i === 3 ? C.green : C.dark, margin: 0 }));
  });
}

// 9. Count and limitations
{
  const slide = pptx.addSlide();
  slideBase(slide, "Conteo general: lectura honesta", "Resultado operativo");
  slide.addText("La Re-ID individual se valida aparte del conteo automático general.", {
    x: 0.8,
    y: 1.4,
    w: 7.4,
    h: 0.28,
    fontSize: 16,
    color: C.muted,
    margin: 0,
  });
  addMetric(slide, "13", "Vacas reales confirmadas", 0.82, 2.0, 2.3, C.green, "referencia visual");
  addMetric(slide, "21", "Etiquetas automáticas", 3.45, 2.0, 2.3, C.clay, "fragmentación del tracker");
  addMetric(slide, "+8", "Error absoluto", 6.08, 2.0, 2.3, C.straw, "sobreconteo documentado");
  slide.addShape(pptx.ShapeType.rect, { x: 0.85, y: 4.05, w: 10.7, h: 0.34, fill: { color: "E6D2A0" }, line: { color: "E6D2A0" } });
  slide.addShape(pptx.ShapeType.rect, { x: 0.85, y: 4.05, w: 6.63, h: 0.34, fill: { color: C.green }, line: { color: C.green } });
  slide.addText("13 reales", { x: 0.98, y: 4.15, w: 1.2, h: 0.12, fontSize: 8.5, bold: true, color: C.white, margin: 0 });
  slide.addText("21 etiquetas", { x: 9.95, y: 4.15, w: 1.0, h: 0.12, fontSize: 8.5, bold: true, color: C.dark, margin: 0 });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.82,
    y: 5.0,
    w: 11.4,
    h: 1.15,
    fill: { color: C.paper },
    line: { color: "E1D0AB", width: 1 },
  });
  addBody(
    slide,
    [
      "Interpretación: las tres vacas catalogadas usan galería específica, embeddings individuales y subembeddings regionales.",
      "Las vacas no catalogadas dependen del tracker general; algunas se fragmentan en más de una etiqueta cuando hay giros, oclusiones o cambios de ángulo.",
    ],
    1.08,
    5.27,
    10.6,
    0.62,
    { fontSize: 12.8, paraSpaceAfterPt: 4 }
  );
}

// 10. Conclusions
{
  const slide = pptx.addSlide();
  slideBase(slide, "Conclusiones y próximos pasos", "Cierre del MVP");
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.72,
    y: 1.5,
    w: 5.65,
    h: 4.8,
    fill: { color: C.green },
    line: { color: C.green },
  });
  slide.addText("El MVP queda completo", {
    x: 1.05,
    y: 1.9,
    w: 4.8,
    h: 0.38,
    fontSize: 24,
    bold: true,
    color: C.white,
    margin: 0,
  });
  addBody(
    slide,
    [
      "• Detector y render HD funcionales",
      "• Re-ID correcta de Marta, Maria y Margarita",
      "• Etiquetas estables para identidades catalogadas",
      "• Métricas calculadas desde reporte JSON",
      "• Limitación de conteo general documentada",
    ],
    1.08,
    2.72,
    4.8,
    2.3,
    { fontSize: 16, color: C.white, paraSpaceAfterPt: 8 }
  );
  slide.addShape(pptx.ShapeType.rect, {
    x: 6.95,
    y: 1.5,
    w: 5.45,
    h: 4.8,
    fill: { color: C.paper },
    line: { color: "E1D0AB", width: 1 },
  });
  slide.addText("Líneas futuras", {
    x: 7.3,
    y: 1.9,
    w: 3.8,
    h: 0.28,
    fontSize: 20,
    bold: true,
    color: C.green,
    margin: 0,
  });
  addBody(
    slide,
    [
      "• Anotar ground truth de bounding boxes para medir mAP de detección.",
      "• Mejorar el conteo de vacas no catalogadas con asociación temporal más fuerte.",
      "• Ampliar la galería a más animales del rodeo.",
      "• Integrar reportes automáticos para uso productivo.",
    ],
    7.3,
    2.52,
    4.65,
    2.35,
    { fontSize: 15, paraSpaceAfterPt: 8 }
  );
  slide.addText("CowTrack demuestra una aplicación concreta de IA al monitoreo ganadero con evidencia visual auditable.", {
    x: 7.3,
    y: 5.42,
    w: 4.65,
    h: 0.42,
    fontSize: 13.5,
    bold: true,
    color: C.brown,
    margin: 0,
    fit: "shrink",
  });
}

async function main() {
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  await pptx.writeFile({ fileName: outPath });
  console.log(outPath);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
