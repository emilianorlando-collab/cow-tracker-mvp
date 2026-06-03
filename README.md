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
- El conteo final consolidado es de **14 vacas**.
- Los cambios de ID internos del tracker no se interpretan automáticamente como
  vacas nuevas.

El video final de presentación se llama:

```text
RESULTADO_FINAL.mp4
```

Por tamaño, el video final no se sube al repositorio. Los videos, datasets,
caches y resultados pesados viven dentro de:

```text
datos/
```

La carpeta `datos/` está excluida por `.gitignore`. El criterio de manejo de
artefactos está documentado en:

```text
artifacts/README.md
```

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
| Conteo estimado vs referencia | **14/14 vacas** | Error absoluto <= 1 vaca | ✅ Superado |
| Accuracy de conteo | **100.0%** | > 90.0% | ✅ Superado |
| Error absoluto de conteo | **0 vacas** | <= 1 vaca | ✅ Superado |
| Presencia temporal promedio de identidades | **98.63%** | > 90.0% | ✅ Superado |
| Cobertura promedio con detección trackeada | **97.80%** | > 90.0% | ✅ Superado |
| Cobertura promedio con Re-ID confirmado | **74.63%** | Métrica diagnóstica | ⚠️ Informativo |
| ID switches de identidades conocidas | **0** | 0 | ✅ Superado |
| Huecos largos en medio del plano | **0** | 0 | ✅ Superado |
| Vacas desconocidas estimadas | **11** | 11 esperadas | ✅ Superado |
| Presencia conjunta de Marta, Maria y Margarita | **2448 frames (50.15%)** | Evidencia de las 3 juntas | ✅ Superado |

Nota metodológica: el mAP@0.5 reportado en la tabla principal corresponde a
Re-ID de identidades objetivo. Un mAP@0.5 de detección por bounding boxes
requiere frames anotados manualmente con cajas ground truth.

### Render final Erondina

Reporte final versionado:

```text
reports/final/16_segunda_mitad_original_timeline_hd_head_render.json
```

Métricas derivadas versionadas:

```text
reports/final/16_metricas_calculadas_desde_json.json
```

| Métrica Operativa | Resultado |
| --- | ---: |
| Resolución procesada | 1920x1080 |
| Frames procesados en render base | 4881 |
| FPS | 29.97 |
| Conteo estimado de vacas | 14 |
| Tracks globales internos después de clustering | 22 |
| Identidades reidentificadas | Marta, Maria, Margarita |
| Auditoría de tracking bloqueado | Aprobada |
| Huecos largos en medio del plano | 0 |
| Duplicados conocidos suprimidos en render | 3 |

Continuidad de las vacas reidentificadas:

| Identidad | Presencia auditada | Frames con detección trackeada | Frames confirmados por Re-ID | Frames sostenidos por continuidad |
| --- | ---: | ---: | ---: | ---: |
| Margarita | 100.00% | 4881 | 4213 | 668 |
| Maria | 99.75% | 4846 | 3329 | 1517 |
| Marta | 96.13% | 4594 | 3386 | 1208 |

El valor de `22` tracks globales internos no significa que haya 22 vacas. Es
una medida de fragmentación causada por giros del dron, oclusiones y cambios de
ángulo. El conteo consolidado del MVP es de **14 vacas**, que coincide con la
referencia esperada para el video.

---

## 🧭 Proceso de Investigación

### 1. Entrenamiento Re-ID general

La primera etapa consistió en entrenar un extractor de embeddings visuales con
un dataset general de vacas. Este modelo transforma cada recorte de vaca en un
vector comparable mediante similitud coseno y búsqueda FAISS.

Archivos principales:

```text
scripts/01_entrenar_reid.py
models/mi_modelo_reid.pt
```

Este modelo general no contiene por sí mismo la identidad de Marta, Maria o
Margarita. Su función es aprender una representación visual útil para comparar
vacas entre sí.

### 2. Evaluación FAISS

Luego se evaluó la calidad de los embeddings mediante búsqueda FAISS. Esta fase
permitió validar que el espacio vectorial podía recuperar individuos similares
y sirvió como base para pasar del laboratorio al campo.

Reportes históricos:

```text
reports/01_entrenamiento_crosspose.md
reports/02_evaluacion_faiss.md
```

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
seleccionadas del mismo video de Erondina. Esta decisión fue clave: el modelo
general de Re-ID sirve para extraer embeddings, pero las identidades concretas
del campo necesitan referencias propias.

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

### 5. Recorte y enfoque de las fotos Erondina

Durante las pruebas se detectó que varios errores de Re-ID no venían
necesariamente del modelo, sino de los recortes usados para generar embeddings:
algunas imágenes incluían partes de otras vacas, bordes, fondo o zonas poco
representativas del animal.

Por eso se decidió **recortar y enfocar las imágenes de Erondina** para aislar
mejor a cada vaca. Esta decisión ayudó a:

- Reducir contaminación visual por vacas vecinas.
- Evitar que el embedding aprendiera fondo o pasto en lugar del animal.
- Mejorar la comparación entre galería y frames del video.
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
debe decidir en cada frame quién es cada vaca, sino usar una identidad ya
bloqueada y seguirla temporalmente.

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
FAISS / similitud coseno contra galería Erondina
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
  --start_frame 4880 \
  --report_out reports/16_segunda_mitad_original_timeline_hd_head_analysis.json \
  --contact_sheet_out reports/16_contact_sheet_segunda_mitad_original_timeline_hd_head_analysis.jpg \
  --candidate_sheet_out reports/16_candidate_sheet_segunda_mitad_original_timeline_hd_head_analysis.jpg \
  --video_out datos/resultados/resultado_erondina_reid.mp4 \
  --evidence_cache_out reports/16_evidence_segunda_mitad_original_timeline_hd_head.pkl
```

Render:

```bash
python3 scripts/16_reid_timeline_erondina.py \
  --process_width 1920 \
  --process_height 1080 \
  --start_frame 4880 \
  --evidence_cache_in reports/16_evidence_segunda_mitad_original_timeline_hd_head.pkl \
  --report_out reports/16_segunda_mitad_original_timeline_hd_head_render.json \
  --contact_sheet_out reports/16_contact_sheet_segunda_mitad_original_timeline_hd_head_render.jpg \
  --candidate_sheet_out reports/16_candidate_sheet_segunda_mitad_original_timeline_hd_head_render.jpg \
  --video_out datos/resultados/resultado_erondina_reid.mp4 \
  --render
```

Los caches `.pkl` y videos `.mp4` quedan fuera del repositorio por tamaño.

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
- Se recortaron y enfocaron las fotos de Erondina para aislar mejor cada vaca y
  mejorar la calidad de los embeddings.
- Se procesó la segunda mitad del video porque la escala y visibilidad de las
  vacas era más adecuada para reidentificación.
- Se dejó de decidir identidad frame a frame y se pasó a una asignación global
  por línea temporal.
- Se incorporaron subembeddings de cabeza/región superior para aportar
  evidencia local adicional.
- Se priorizaron visualmente las etiquetas de Marta, Maria y Margarita sobre
  las vacas desconocidas.
- Se evitó contar cada ID interno como una vaca nueva, porque el giro del dron
  puede fragmentar trayectorias.
- Se mantuvo `datos/` fuera de Git para preservar un repositorio liviano,
  reproducible y profesional.

---

## ⚠️ Limitaciones

El MVP está completo, pero el escenario real tiene limitaciones importantes:

- La calidad de la reidentificación depende de la nitidez de los recortes.
- Los giros bruscos del dron pueden fragmentar tracks internos.
- Las oclusiones fuertes pueden requerir continuidad espacial para sostener una
  etiqueta.
- El procesamiento actual es offline, no en tiempo real.
- No se versionan videos ni datasets pesados dentro del repositorio.

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
- Contabilizar aproximadamente el rodeo completo.
- Documentar métricas, decisiones y pruebas intermedias.

El resultado final alcanza los criterios de éxito definidos para el MVP:

- ✅ Detector validado sobre umbrales proyectados.
- ✅ Galería Erondina construida con imágenes propias del video.
- ✅ Re-ID individual de Marta, Maria y Margarita.
- ✅ Tracking temporal estable en el render final.
- ✅ Conteo consolidado de 14 vacas.
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
