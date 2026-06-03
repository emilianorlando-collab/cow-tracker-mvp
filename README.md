# CowTrack MVP

Sistema de detección, seguimiento y reidentificación individual de vacas en
video aéreo mediante YOLOv8, embeddings profundos, FAISS y tracking temporal.

El MVP está orientado a videos reales de campo capturados con dron. Su objetivo
no es solamente contar animales visibles, sino sostener identidades individuales
cuando existen vacas previamente catalogadas.

**Estado del proyecto:** MVP completo.

## Objetivo

El proyecto busca resolver tres tareas principales:

1. Detectar vacas en video mediante un modelo YOLO entrenado para el dominio.
2. Generar embeddings visuales de cada vaca para reidentificación individual.
3. Mantener el tracking temporal de cada animal a lo largo del video, evitando
   cambios de identidad cuando la cámara gira, la vaca se mueve o existen
   oclusiones parciales.

La versión final del MVP fue validada sobre el video de campo Erondina, con
tres vacas catalogadas:

| Identidad | Característica visual usada para auditoría |
| --- | --- |
| Marta | Vaca castaña |
| Maria | Vaca negra |
| Margarita | Vaca castaña |

Las etiquetas de estas tres vacas tienen prioridad visual en el video final. Si
una caja de una vaca desconocida se superpone con una vaca reidentificada, se
prioriza la etiqueta de Marta, Maria o Margarita.

## Resultado Final Erondina

El resultado final del caso Erondina genera un video renderizado en HD donde:

- Marta, Maria y Margarita aparecen con etiquetas visibles y persistentes
  mientras permanecen dentro del plano.
- Las vacas no catalogadas se detectan y se contabilizan como parte del rodeo.
- El conteo estimado final es de 14 vacas.
- El render evita que identidades internas fragmentadas del tracker se
  interpreten automáticamente como animales nuevos.

El video final de presentación se llama:

```text
RESULTADO_FINAL.mp4
```

Por su tamaño, el video no se versiona en Git. Los videos, datasets, caches y
artefactos pesados viven dentro de `datos/`, carpeta excluida mediante
`.gitignore`. La ubicación y criterio de manejo de artefactos están documentados
en:

```text
artifacts/README.md
```

El índice profesional de scripts, incluyendo las pruebas intermedias realizadas
durante la investigación, está disponible en:

```text
docs/SCRIPTS_INTERMEDIOS.md
```

## Proceso de Investigación

### 1. Entrenamiento Re-ID general

La primera etapa consistió en entrenar un extractor de embeddings con un dataset
general de vacas. Ese modelo base permite transformar cada recorte de vaca en
un vector visual comparable mediante similitud coseno o búsqueda FAISS.

El entrenamiento original se apoya en:

```text
scripts/01_entrenar_reid.py
models/mi_modelo_reid.pt
```

El modelo Re-ID general no identifica automáticamente a Marta, Maria o
Margarita. Su función es producir embeddings útiles para comparar individuos.
Para reconocer vacas específicas del campo Erondina fue necesario construir una
galería propia con imágenes de esas tres vacas tomadas del mismo video real.

### 2. Evaluación FAISS y validación de embeddings

Luego se evaluó la capacidad del modelo para distinguir individuos mediante
FAISS. Esta fase permitió validar que el espacio de embeddings era útil para
comparar vacas, pero también mostró que en condiciones reales de dron aparecen
problemas adicionales: distancia, ángulo, baja resolución, cambios de pose,
oclusiones y giro de cámara.

Reportes históricos:

```text
reports/01_entrenamiento_crosspose.md
reports/02_evaluacion_faiss.md
```

### 3. Detección y tracking base

La etapa de video combina detección YOLO con tracking temporal. El pipeline
inicial permitía detectar vacas y mantener IDs internos, pero en el video real
se observaron problemas típicos de campo:

- Fragmentación de una misma vaca en múltiples IDs internos.
- Cambios de identidad cuando el dron giraba.
- Detecciones faltantes en frames aislados.
- Superposición entre vacas conocidas y desconocidas.

Estos problemas motivaron la etapa final offline basada en línea temporal.

### 4. Galería Erondina

Para el caso final se crearon carpetas con fotos manualmente seleccionadas de
Marta, Maria y Margarita desde el propio video de Erondina. Esas imágenes se
usaron para generar una galería específica del campo:

```text
models/erondina_gallery_embeddings_enfocada_filtrada.npz
```

La galería contiene embeddings de referencia y prototipos por identidad:

- `gallery_vectors`
- `gallery_labels`
- `gallery_paths`
- `proto_vectors`
- `proto_labels`

Esta decisión separa claramente el modelo general de Re-ID de los embeddings de
las vacas puntuales que se quieren reconocer en Erondina.

### 5. Análisis offline antes del render

Antes de renderizar el video final se ejecutó una auditoría previa sobre la
segunda mitad del video, donde las vacas se observan con mejor escala y
visibilidad. El análisis agrupa fragmentos de tracking, compara embeddings
contra la galería Erondina y decide que identidad global corresponde a Marta,
Maria y Margarita.

La lógica final está implementada en:

```text
scripts/16_reid_timeline_erondina.py
```

Este script realiza:

- Detección de vacas con YOLO y tracking base.
- Extracción de embeddings de cuerpo.
- Extracción auxiliar de región superior/cabeza aproximada.
- Agrupación de fragmentos en identidades globales.
- Asignación de identidades conocidas mediante la galería Erondina.
- Auditoría de continuidad antes del render.
- Render final con etiquetas priorizadas.

El uso de color fue considerado como criterio de auditoría visual del resultado,
no como una regla dura de identificación. El pipeline final decide las
identidades mediante embeddings, consistencia temporal y continuidad espacial.

## Métricas Finales

Reporte final versionado:

```text
reports/final/16_segunda_mitad_original_timeline_hd_head_render.json
```

Métricas principales del render final:

| Métrica | Resultado |
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

El valor de `22` tracks globales internos no representa 22 vacas reales. Es una
medida de fragmentación del tracking producida por cambios de ángulo, giros del
dron y oclusiones. Para el conteo final se utiliza la estimación consolidada de
vacas visibles, que coincide con la referencia esperada de 14 animales.

## Estructura del Repositorio

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

## Ejecución

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

## Decisiones Técnicas Relevantes

- Se separó el modelo Re-ID general de la galería específica de Erondina.
- Se eligieron fotos del propio video para reducir diferencia de dominio entre
  entrenamiento puntual y render final.
- Se procesó la segunda mitad del video porque la escala de las vacas es más
  adecuada para reidentificación visual.
- Se agregó una auditoría previa al render para verificar continuidad de Marta,
  Maria y Margarita.
- Se priorizaron visualmente las etiquetas de vacas reidentificadas sobre las
  vacas desconocidas.
- Se consideró la fragmentación por giro de dron para evitar que cada ID
  interno se cuente como una vaca distinta.
- Se mantuvo `datos/` fuera de Git para preservar un repositorio liviano y
  profesional.

## Limitaciones

El MVP está completo, pero existen límites propios del escenario:

- La reidentificación depende de la calidad de los recortes y de la visibilidad
  real de cada vaca.
- Cambios bruscos de ángulo del dron pueden fragmentar tracks internos.
- Oclusiones fuertes pueden requerir continuidad espacial para sostener la
  etiqueta entre detecciones confirmadas.
- El procesamiento actual es offline; no está optimizado para tiempo real.

Estas limitaciones no impiden el cierre del MVP, pero son puntos naturales para
una etapa posterior de investigación.

## Conclusiones

El MVP demuestra un pipeline completo para detección, conteo, tracking y
reidentificación individual de ganado en un video real de campo.

El resultado final logra identificar y sostener las tres vacas catalogadas
durante la mayor parte del video procesado, prioriza sus etiquetas en el render
y mantiene un conteo consolidado de 14 vacas. La solución también documenta los
problemas encontrados en campo, especialmente la fragmentación causada por el
movimiento del dron, y los aborda mediante análisis offline, embeddings propios
de Erondina y auditoría de continuidad.

Con este resultado, el proyecto queda en estado de MVP completo: existe un
flujo reproducible, reportes técnicos, métricas finales, artefactos
intermedios y un video final listo para presentación.

## Tecnologías

- YOLOv8 / Ultralytics
- PyTorch
- Torchvision / ResNet18
- FAISS
- OpenCV
- NumPy
- SciPy

## Autor

Emiliano Orlando
