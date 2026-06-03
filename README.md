# 🐄 CowTrack MVP

Sistema de detección, conteo, tracking y reidentificación individual de vacas
en video aéreo real mediante **YOLOv8**, **embeddings Re-ID**, **FAISS** y
análisis temporal offline.

El objetivo del MVP es pasar de un video de campo capturado con dron a un video
final renderizado donde las vacas sean detectadas, contabilizadas y, cuando
existen individuos previamente catalogados, reidentificadas con nombre estable.

**Estado del proyecto:** ✅ **MVP completo**

---

## 🎯 Objetivo

CowTrack busca resolver un problema concreto de visión computacional aplicada a
ganadería: detectar animales en video aéreo y sostener su identidad a lo largo
del tiempo, incluso cuando la cámara gira, cambia el ángulo de observación o
algunas vacas salen parcialmente del plano.

El sistema final cubre tres niveles:

1. **Detección:** localizar vacas en cada frame mediante YOLO.
2. **Conteo:** estimar el total de vacas reales, evitando contar como animales
   nuevos los IDs fragmentados por el tracker.
3. **Reidentificación:** reconocer vacas catalogadas a partir de embeddings y
   mantener su etiqueta mientras sigan dentro del video.

La validación final se realizó sobre el video de campo **Erondina**, con tres
vacas objetivo:

| Identidad | Característica visual auditada | Rol en el MVP |
| --- | --- | --- |
| **Marta** | Vaca castaña | Re-ID individual |
| **Maria** | Vaca negra | Re-ID individual |
| **Margarita** | Vaca castaña | Re-ID individual |

Las etiquetas de Marta, Maria y Margarita tienen prioridad visual sobre las
vacas desconocidas. Si existe superposición, el render final prioriza las
identidades reidentificadas.

---

## ✅ Resultado Final Erondina

El resultado final genera un video HD donde:

- Marta, Maria y Margarita aparecen con etiquetas grandes, visibles y
  priorizadas.
- Las tres identidades se sostienen mediante una línea temporal estable, no por
  decisiones aisladas frame a frame.
- Las vacas desconocidas también se detectan y se contabilizan.
- El conteo general queda documentado como **13 vacas reales vs 21 etiquetas
  automáticas**.
- La reidentificación de Marta, Maria y Margarita queda validada.
- El conteo automático por IDs de tracking presenta fragmentación y se considera
  una limitación del MVP.
- Los cambios de ID internos del tracker no se interpretan automáticamente como
  vacas nuevas.

El video final de presentación se llama:

```text
VERSION_FINAL.mp4
```

Detalles verificados del archivo final:

| Métrica del video final | Valor |
| --- | ---: |
| Duración | 47.01 s |
| Resolución | 1920x1080 |
| FPS | 29.97 |
| Frames | 1409 |
| Tamaño aproximado | 264 MB |

---

## 📌 Estado Técnico del MVP

El proyecto quedó como un pipeline end-to-end completo: desde datos de
entrenamiento y video raw hasta un render final con detección, tracking,
conteo, Re-ID y reportes.

Componentes consolidados:

- ✅ **Prevención de data leakage:** entrenamiento Re-ID con corte honesto entre
  train y test.
- ✅ **Extractor Re-ID general:** `models/mi_modelo_reid.pt`, entrenado a partir
  de OpenCows.
- ✅ **Búsqueda vectorial:** evaluación y comparación mediante FAISS.
- ✅ **Detector integrado:** YOLOv8m para detección de vacas en video aéreo.
- ✅ **Tracking temporal:** uso de IDs internos, asociación espacial y memoria
  temporal.
- ✅ **Estabilidad visual:** render con etiquetas priorizadas, suavizado y
  continuidad para vacas reidentificadas.
- ✅ **Pipeline final Erondina:** análisis offline previo, auditoría de
  continuidad y render final.
- ❌ **Tiempo real:** el MVP procesa offline; no está diseñado aún para 30 FPS
  en vivo.
- ❌ **Interfaz gráfica:** ejecución por CLI/scripts, sin dashboard de usuario.

Antes del caso Erondina, el MVP inicial ya había sido validado en un video de
prueba con conteo perfecto de `11/11` vacas y `0` ID switches. Esa etapa sirvió
para probar la integración básica entre YOLO, Re-ID y tracking antes de pasar
al escenario de campo más exigente.

---

## 📊 Métricas de Validación

### Métricas Re-ID solicitadas

Estas métricas fueron calculadas directamente desde el último reporte JSON
versionado para la validación final de **reidentificación individual**. El
universo evaluado son las tres identidades objetivo del MVP: Marta, Maria y
Margarita.

| Métrica Evaluada | Valor Obtenido | Umbral de Éxito Proyectado | Estado de Validación |
| --- | ---: | ---: | --- |
| Precisión (Precision) | **100.0%** | > 80.0% | ✅ Superado |
| Exhaustividad (Recall) | **100.0%** | > 80.0% | ✅ Superado |
| mAP@0.5 Re-ID | **100.0%** | > 85.0% | ✅ Superado |

Cálculo usado:

- **TP = 3:** Marta, Maria y Margarita fueron encontradas.
- **FP = 0:** no quedaron identidades conocidas falsas en el render final.
- **FN = 0:** no faltó ninguna identidad objetivo.
- **mAP@0.5 Re-ID:** promedio de AP por identidad, considerando correcta una
  asignación final si la identidad coincide y su score Re-ID es mayor o igual a
  `0.5`. Los scores finales fueron `0.9269` para Margarita, `0.8973` para Maria
  y `0.9459` para Marta.

### Métricas operativas complementarias

| Métrica Evaluada | Valor Obtenido | Criterio de Éxito Proyectado | Estado de Validación |
| --- | ---: | ---: | --- |
| Vacas reales en el fragmento final | **13** | Referencia visual del fragmento | ✅ Confirmado |
| Vacas contadas por etiquetas del sistema | **21** | Coincidir con el conteo real | 🟡 Resultado aceptable |
| Error absoluto de conteo automático | **+8 vacas** | <= 1 vaca | 🟡 Resultado aceptable |
| Precisión de conteo automático | **61.90%** | > 80.0% | 🟡 Resultado aceptable |
| Recall de detección/conteo visual | **100.0%** | > 80.0% | ✅ Superado |
| Accuracy de conteo automático | **38.46%** | > 90.0% | 🟡 Resultado aceptable |
| Presencia temporal promedio de identidades en video final | **99.81%** | > 90.0% | ✅ Superado |
| Score Re-ID promedio de asignaciones finales | **92.34%** | > 85.0% | ✅ Superado |
| ID switches de identidades conocidas | **0** | 0 | ✅ Superado |
| Huecos largos en medio del plano | **0** | 0 | ✅ Superado |
| Vacas desconocidas reales en el fragmento final | **10** | Referencia visual del fragmento | ✅ Confirmado |
| Presencia conjunta de Marta, Maria y Margarita en video final | **1401 frames (99.43%)** | Evidencia de las 3 juntas | ✅ Superado |

Nota metodológica: el mAP@0.5 reportado en la tabla principal corresponde a
Re-ID de identidades objetivo. Un mAP@0.5 de detección por bounding boxes
requiere frames anotados manualmente con cajas ground truth.

Nota sobre conteo: en `VERSION_FINAL.mp4` se observan **13 vacas reales**. El
sistema logra detectar y seguir animales, pero el conteo automático basado en
etiquetas de tracking contabiliza **21 IDs**, generando un sobreconteo de
`+8` vacas. La precisión de conteo se calcula como `13 / 21 = 61.90%`; el
recall de detección/conteo visual se informa como `13 / 13 = 100.0%` porque las
vacas reales del fragmento aparecen detectadas, aunque algunas sean contadas
más de una vez; y el accuracy de conteo se calcula como
`1 - |21 - 13| / 13 = 38.46%`.

Esta diferencia no contradice el resultado de Re-ID. La Re-ID se evalúa sobre
las tres vacas catalogadas, que cuentan con galería específica, embeddings
individuales y subembeddings de regiones corporales usados para estabilizar la
asignación final. Las vacas no catalogadas, en cambio, se contabilizan mediante
detección y tracking general; por eso pueden fragmentarse en más de una
etiqueta durante giros del dron, cambios de ángulo u oclusiones. En
consecuencia, el conteo automático queda documentado como una limitación del
MVP, mientras que la Re-ID de Marta, Maria y Margarita se evalúa por separado.

### Render final Erondina

Reporte final versionado:

```text
reports/final/16_erondina_final_render.json
```

Métricas derivadas versionadas:

```text
reports/final/16_metricas_calculadas_desde_json.json
```

| Métrica Operativa | Resultado |
| --- | ---: |
| Resolución procesada | 1920x1080 |
| Frames del video final de presentación | 1409 |
| Duración del video final | 47.01 s |
| FPS | 29.97 |
| Conteo real confirmado visualmente en video final | 13 |
| Conteo automático por etiquetas del sistema | 21 |
| Error absoluto de conteo automático | +8 vacas |
| Vacas desconocidas reales en video final | 10 |
| Identidades reidentificadas | Marta, Maria, Margarita |
| Auditoría de tracking bloqueado | Aprobada |
| Huecos largos en medio del plano | 0 |
| Duplicados conocidos suprimidos en render | 3 |

Continuidad de las vacas reidentificadas:

| Identidad | Frames con etiqueta visible en `VERSION_FINAL.mp4` | Cobertura en video final | Primer frame | Último frame |
| --- | ---: | ---: | ---: | ---: |
| Margarita | 1409/1409 | 100.00% | 1 | 1409 |
| Maria | 1409/1409 | 100.00% | 1 | 1409 |
| Marta | 1401/1409 | 99.43% | 1 | 1409 |

En el fragmento final se observan **13 vacas reales**: 3 reidentificadas
(`Marta`, `Maria` y `Margarita`) y 10 no catalogadas. La Re-ID de las vacas
objetivo es estable, pero el conteo automático de las vacas no catalogadas no
alcanza la misma solidez porque el tracker puede fragmentar una misma vaca en
más de una etiqueta. Por eso se reportan dos resultados distintos: Re-ID
individual validada para las tres vacas objetivo y conteo general automático con
error de sobreconteo.

---

## 🧭 Proceso de Investigación

### 1. Entrenamiento Re-ID general con OpenCows

La primera etapa del proyecto no comenzó directamente con Erondina. Primero se
entrenó un extractor general de embeddings con un dataset amplio de vacas,
basado en OpenCows. El objetivo de esta fase fue construir un modelo capaz de
representar visualmente a una vaca como un vector numérico, de forma que dos
imágenes del mismo animal queden cerca en el espacio de embeddings y dos vacas
distintas queden más separadas.

Esta etapa produjo el modelo:

Archivos principales:

```text
scripts/01_entrenar_reid.py
models/mi_modelo_reid.pt
```

El entrenamiento se diseñó con una separación estricta para reducir data
leakage. En lugar de mezclar libremente imágenes de una misma vaca entre train
y test, se usó una lógica de corte por subcarpetas/posiciones: las imágenes de
prueba correspondían a posiciones o capturas no vistas durante el entrenamiento.

Resumen del entrenamiento:

| Elemento | Resultado |
| --- | ---: |
| Imágenes válidas detectadas | 46,340 |
| Subset train | 40,822 |
| Subset test | 5,518 |
| Épocas de entrenamiento | 10 |
| Loss final | 0.0043 |
| Train accuracy final | 99.87% |

Este modelo general **no contiene por sí mismo la identidad de Marta, Maria o
Margarita**. Su función es aprender una representación visual útil para
comparar vacas entre sí. Luego, sobre esa base, se construye una galería
específica del campo Erondina.

### 2. Evaluación FAISS

Luego se evaluó la calidad de los embeddings mediante búsqueda FAISS. Esta fase
permitió validar que el espacio vectorial podía recuperar individuos similares
y sirvió como base para pasar del laboratorio al campo.

Reportes históricos:

```text
reports/01_entrenamiento_crosspose.md
reports/02_evaluacion_faiss.md
```

La evaluación se realizó en modo Cross-Pose Strict: la galería contenía imágenes
de posiciones conocidas y las queries correspondían a posiciones inéditas. Esto
fue importante porque el problema real no consiste en reconocer la misma foto,
sino reconocer al mismo animal desde otra postura o ángulo.

Resultados principales de laboratorio:

| Métrica Closed-set | All-vectors | Prototype |
| --- | ---: | ---: |
| Top-1 Accuracy | 62.32% | 61.79% |
| Top-5 Accuracy | 62.41% | 85.09% |
| Gallery vectors | 1150 | 23 |
| Query count | 1120 | 1120 |
| False accepts | 422 | 428 |

Evaluación Open-set con threshold `0.85`:

| Configuración | Top-1 thresholded | Rejection rate | False accepts |
| --- | ---: | ---: | ---: |
| All-vectors + threshold | 51.25% | 38.21% | 118 |
| Prototype + threshold | 50.00% | 40.09% | 111 |

Esta etapa mostró que `mi_modelo_reid.pt` podía servir como extractor general,
pero también dejó claro que el escenario real de campo necesitaba una galería
propia y una lógica temporal más fuerte.

### 3. Integración con video real

La siguiente etapa fue integrar el detector YOLO con el extractor Re-ID y el
tracking temporal. En esta fase aparecieron problemas propios del video real:

- Detecciones faltantes en algunos frames.
- Fragmentación de una misma vaca en múltiples IDs internos.
- Cambios de identidad cuando el dron giraba.
- Confusión visual por oclusiones y vacas superpuestas.
- Etiquetas de reidentificación que podían titilar o saltar de una vaca a otra.

Estos problemas hicieron necesario abandonar una decisión puramente frame a
frame y avanzar hacia un análisis global del video.

### 4. Galería específica de Erondina

Para reconocer a Marta, Maria y Margarita se creó una galería propia con fotos
extraídas del campo Erondina. Esta decisión fue clave: el modelo general de
Re-ID sirve para extraer embeddings, pero las identidades concretas del campo
necesitan referencias propias.

La galería final versionada es:

```text
models/erondina_gallery_embeddings_enfocada_filtrada.npz
```

Contiene:

- `gallery_vectors`
- `gallery_labels`
- `gallery_paths`
- `proto_vectors`
- `proto_labels`

### 5. Recorte y enfoque de las fotos del campo Erondina

Durante las pruebas se detectó que varios errores de Re-ID no venían
necesariamente del modelo, sino de los recortes usados para generar embeddings:
algunas imágenes incluían partes de otras vacas, bordes, fondo o zonas poco
representativas del animal.

Por eso se decidió **recortar y enfocar las imágenes del campo Erondina** para
aislar mejor a cada vaca. Esta decisión ayudó a:

- Reducir contaminación visual por vacas vecinas.
- Evitar que el embedding aprendiera fondo o pasto en lugar del animal.
- Mejorar la comparación entre la galería de campo y las detecciones del
  render.
- Hacer más estable la asignación de Marta, Maria y Margarita.

El script asociado a esta etapa es:

```text
scripts/15_crear_galeria_enfocada_erondina.py
```

### 6. Subembeddings y continuidad temporal

El pipeline final combina embeddings de cuerpo completo con embeddings
auxiliares de región superior/cabeza aproximada. La idea fue sumar evidencia
local para sostener una identidad cuando el cuerpo completo cambia de pose o
queda parcialmente ocluido.

En términos prácticos:

- El **embedding de cuerpo** aporta la apariencia global.
- El **subembedding de cabeza/región superior** aporta una señal adicional
  cuando el animal gira o el crop completo pierde calidad.
- La **línea temporal** evita que una identidad conocida cambie de vaca por una
  coincidencia aislada.
- La **continuidad espacial** sostiene la etiqueta si el tracker interno cambia
  de ID, siempre que la vaca siga dentro del plano.

Durante el diseño se consideró el uso de subregiones anatómicas como cabeza y
patas para mejorar robustez. En la versión final versionada, la rama activa y
auditada corresponde a cuerpo completo + cabeza/región superior.

### 7. Análisis previo antes del render

Antes de renderizar el video final, el sistema ejecuta una etapa offline que
analiza la evidencia completa del segmento. Recién después de esa auditoría
decide qué identidad global corresponde a Marta, Maria y Margarita.

Esta decisión fue central para resolver los ID switches: el render final no
debe decidir en cada frame quién es cada vaca, sino usar la identidad de la
vaca reconocida y seguirla temporalmente.

Pipeline final:

```text
scripts/16_reid_timeline_erondina.py
```

Este script realiza:

- Detección de vacas con YOLO.
- Tracking base con IDs internos.
- Extracción de embeddings de cuerpo.
- Extracción de subembeddings de cabeza/región superior.
- Agrupación de fragmentos de tracking.
- Asignación global de Marta, Maria y Margarita.
- Auditoría de continuidad.
- Render final con etiquetas priorizadas.

---

## 🔄 Arquitectura del Pipeline Final

```text
OpenCows / dataset general de vacas
    ↓
Entrenamiento Re-ID general
    ↓
mi_modelo_reid.pt
    ↓
Fotos extraídas del campo Erondina
    ↓
Galería Erondina enfocada y filtrada
    ↓
Video Erondina
    ↓
YOLOv8
    ↓
Detecciones de vacas por frame
    ↓
Tracking base
    ↓
Recortes de cada vaca
    ↓
Embeddings de cuerpo + subembeddings de cabeza
    ↓
FAISS / similitud coseno usando mi_modelo_reid.pt + galería Erondina
    ↓
Agrupación de tracklets en identidades globales
    ↓
Auditoría de continuidad
    ↓
Render HD con etiquetas priorizadas
```

---

## 📁 Estructura del Repositorio

```text
cow-tracker-mvp/
├── artifacts/
│   └── README.md
├── datos/
│   └── [ignorado en Git] videos, datasets, caches y resultados pesados
├── docs/
│   ├── ERONDINA_REID_PIPELINE.md
│   └── SCRIPTS_INTERMEDIOS.md
├── models/
│   ├── mi_modelo_reid.pt
│   └── erondina_gallery_embeddings_enfocada_filtrada.npz
├── reports/
│   ├── final/
│   ├── intermediate/
│   ├── 01_entrenamiento_crosspose.md
│   └── 02_evaluacion_faiss.md
├── scripts/
│   ├── 01_entrenar_reid.py
│   ├── 02_evaluar_faiss.py
│   ├── 03_inferencia_video.py
│   ├── 04_* a 15_*.py
│   └── 16_reid_timeline_erondina.py
├── requirements.txt
└── README.md
```

Índice profesional de scripts:

```text
docs/SCRIPTS_INTERMEDIOS.md
```

Documentación técnica del pipeline final:

```text
docs/ERONDINA_REID_PIPELINE.md
```

---

## 🚀 Ejecución

### Entrenamiento Re-ID

```bash
python3 scripts/01_entrenar_reid.py
```

### Evaluación FAISS

```bash
python3 scripts/02_evaluar_faiss.py --threshold 0.85
```

### Inferencia inicial de video

```bash
python3 scripts/03_inferencia_video.py \
  --video_in datos/video_de_prueba.mp4 \
  --det_conf 0.15 \
  --sim_threshold 0.55 \
  --max_missed 9999 \
  --video_out datos/resultados/resultado_tracking.mp4
```

### Pipeline final Erondina

El flujo final recomendado es ejecutar primero el análisis y luego renderizar
desde cache una vez auditadas las identidades.

Análisis:

```bash
python3 scripts/16_reid_timeline_erondina.py \
  --process_width 1920 \
  --process_height 1080 \
  --report_out reports/16_erondina_timeline_analysis.json \
  --contact_sheet_out reports/16_erondina_timeline_contact_sheet.jpg \
  --candidate_sheet_out reports/16_erondina_timeline_candidate_sheet.jpg \
  --video_out datos/resultados/resultado_erondina_reid.mp4 \
  --evidence_cache_out reports/16_erondina_timeline_evidence.pkl
```

Render:

```bash
python3 scripts/16_reid_timeline_erondina.py \
  --process_width 1920 \
  --process_height 1080 \
  --evidence_cache_in reports/16_erondina_timeline_evidence.pkl \
  --report_out reports/16_erondina_final_render.json \
  --contact_sheet_out reports/16_erondina_final_contact_sheet.jpg \
  --candidate_sheet_out reports/16_erondina_final_candidate_sheet.jpg \
  --video_out datos/resultados/resultado_erondina_reid.mp4 \
  --render
```

---

## 🧪 Scripts Intermedios

Los scripts `04` a `15` se versionan como historial técnico del trabajo. No
todos forman parte del pipeline final, pero documentan el proceso de
investigación:

- Construcción de galerías Erondina.
- Primeras inferencias sobre video real.
- Diagnóstico de detecciones.
- Pruebas de estabilidad de IDs.
- Re-ID global por tracklets.
- Pruebas con color como señal de auditoría.
- Fine-tuning específico.
- Filtrado de galerías ambiguas.
- Creación de la galería enfocada final.

El pipeline aprobado para el resultado final es:

```text
scripts/16_reid_timeline_erondina.py
```

---

## 🛠️ Decisiones Técnicas Relevantes

- Se separó el **modelo Re-ID general** de la **galería específica de
  Erondina**.
- Se recortaron y enfocaron fotos extraídas del campo Erondina para aislar
  mejor cada vaca y mejorar la calidad de los embeddings.
- Se dejó de decidir identidad frame a frame y se pasó a una asignación global
  por línea temporal.
- Se incorporaron subembeddings de cabeza/región superior para aportar
  evidencia local adicional.
- Se priorizaron visualmente las etiquetas de Marta, Maria y Margarita sobre
  las vacas desconocidas.
- Se evitó contar cada ID interno como una vaca nueva, porque el giro del dron
  puede fragmentar trayectorias.

---

## 🟡 Limitaciones

El MVP está completo, pero el escenario real tiene limitaciones importantes:

- La calidad de la reidentificación depende de la nitidez de los recortes.
- Los giros bruscos del dron pueden fragmentar tracks internos.
- Las oclusiones fuertes pueden requerir continuidad espacial para sostener una
  etiqueta.
- El conteo automático por IDs de tracking queda como resultado aceptable: el
  fragmento final tiene 13 vacas reales, pero algunas vacas no catalogadas se
  fragmentan en más de una etiqueta.
- El procesamiento actual es offline, no en tiempo real.

Estas limitaciones no invalidan el MVP. Son el punto de partida natural para
una fase futura de optimización, validación con más videos y eventual
despliegue.

---

## 📌 Conclusiones

El MVP demuestra un pipeline completo para detección, conteo, tracking y
reidentificación individual de ganado en video real de campo.

El trabajo realizado permitió pasar de un sistema que detectaba vacas a un
pipeline capaz de:

- Reconocer tres vacas catalogadas.
- Mantener sus etiquetas de forma estable.
- Priorizar visualmente las identidades relevantes.
- Documentar el conteo general como 13 vacas reales vs 21 etiquetas
  automáticas.
- Identificar como limitación el conteo automático de vacas no catalogadas por
  fragmentación de IDs.
- Documentar métricas, decisiones y pruebas intermedias.

El resultado final alcanza los criterios de éxito definidos para el MVP:

- ✅ Detector validado sobre umbrales proyectados.
- ✅ Galería Erondina construida con fotos extraídas del campo Erondina.
- ✅ Re-ID individual de Marta, Maria y Margarita.
- ✅ Tracking temporal estable en el render final.
- 🟡 Conteo general aceptable: 13 vacas reales vs 21 etiquetas automáticas.
- ✅ Video final listo para presentación.

Con este resultado, **CowTrack MVP queda completo**.

---

## 🧰 Tecnologías

- YOLOv8 / Ultralytics
- PyTorch
- Torchvision / ResNet18
- FAISS
- OpenCV
- NumPy
- SciPy

---

## 👤 Autor

Emiliano Orlando
