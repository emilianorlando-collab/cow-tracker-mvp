# 🐄 CowTrack MVP - Sistema Híbrido de Re-ID y Tracking de Ganado

**Identificación individual y seguimiento espacial de vacas en video usando YOLOv8 + ResNet18 + FAISS.**

## 🎯 Objetivo del Proyecto

Crear un sistema de visión computacional robusto capaz de:
1. **Detectar** vacas en videos aéreos (dron) de alta resolución.
2. **Identificar** biométricamente a cada individuo (Re-ID) usando embeddings profundos.
3. **Rastrear (Tracking)** a las vacas a lo largo del video sin perder su identidad, combinando memoria espacial (anclaje) y similitud visual.

**Status actual:** MVP Funcional "End-to-End". Validado en video de prueba con **100% de precisión en conteo (11/11 vacas)** y **0 ID Switches**.

---

## 📊 Estado actual del proyecto

✅ **Pipeline End-to-End completo:** Desde el video raw hasta el video renderizado con IDs.
✅ **Prevención de Data Leakage:** Entrenamiento Re-ID con "Corte Honesto" (80% inicial para train, 20% inédito para test).
✅ **Detector Integrado:** YOLOv8m en Alta Definición (1280px) con NMS manual y filtro "Anti-Mega-Cajas".
✅ **Tracker Híbrido Universal:** Matriz de costos que combina Distancia Espacial + Similitud Coseno (FAISS).
✅ **Estabilidad Visual:** Cajas "pegajosas" (Sticky Bounding Boxes) con memoria infinita y suavizado EMA.
❌ **Validación en Tiempo Real:** Actualmente procesa offline (post-procesamiento).
❌ **Interfaz de Usuario:** Ejecución exclusiva por CLI (Terminal).

---

## 📁 Estructura del Proyecto

```text
cow-tracker-mvp/
├── datos/                  # [IGNORADO EN GIT] Videos raw y fotos de entrenamiento
├── models/                 # [IGNORADO EN GIT] Pesos de YOLO (.pt) y Re-ID (.pt)
├── resultados/             # [IGNORADO EN GIT] Videos finales renderizados
├── reports/                # Reportes operativos y métricas de FAISS (.txt, .md)
├── scripts/
│   ├── 01_entrenar_reid.py    # Entrenamiento ResNet18 con Corte Honesto
│   ├── 02_evaluar_faiss.py    # Inferencia y métricas Cross-Pose FAISS
│   └── 03_inferencia_video.py # Pipeline End-to-End (YOLO + Tracker Híbrido)
├── .gitignore              # Escudo para evitar subir archivos pesados
├── README.md               # Este archivo
└── requirements.txt        # Dependencias de Python
```

---

## 🚀 Quickstart (Uso rápido por CLI)

### 1. Entrenamiento del Extractor Re-ID
Entrena el modelo ResNet18 asegurando que las imágenes de prueba sean posiciones inéditas (Cross-Pose):
```bash
python3 scripts/01_entrenar_reid.py
```

### 2. Evaluación de Laboratorio (FAISS)
Evalúa el modelo generado contra una galería cerrada y abierta:
```bash
python3 scripts/02_evaluar_faiss.py --threshold 0.85
```

### 3. Inferencia de Video en Producción (Tracking)
Procesa un video aéreo, aplica el tracker híbrido y exporta el video renderizado con métricas:
```bash
python3 scripts/03_inferencia_video.py --video_in datos/video_de_prueba.mp4 --det_conf 0.15 --sim_threshold 0.55 --max_missed 9999 --video_out resultados/resultado_final.mp4 | tee reports/03_tracking_final.txt
```

---

## 📈 Resultados Finales del MVP

### Fase 1: Entrenamiento (Data Set Masivo)
Se utilizó un dataset de **46,340 imágenes válidas**, separado de forma estricta (temporal/secuencial) para evitar fugas de información.
* **Subset Train:** 40,822 imágenes
* **Subset Test:** 5,518 imágenes
* **Loss Final:** 0.0043 | **Train Accuracy:** 99.87%

### Fase 2: Evaluación Cross-Pose (FAISS)
Métricas calculadas sobre **1120 imágenes inéditas** contra una galería de 1150 imágenes conocidas. El modelo logra un 62% de precisión estricta bajo condiciones severas de cambio de postura (Cross-Pose Strict).

| Métrica Closed-set | All-vectors | Prototype |
| :--- | :--- | :--- |
| **Top-1 Accuracy** | 62.32% | 61.79% |
| **Top-5 Accuracy** | 62.41% | 85.09% |
| False accepts | 422 | 428 |

### Fase 3: Tracking en Video (El Mundo Real)
El algoritmo híbrido (YOLOv8m + Anclaje Espacial + FAISS) demostró un rendimiento comercial en el video de prueba.

| Métrica Operativa de Video | Resultado | Interpretación |
| :--- | :--- | :--- |
| **IDs Únicos Consolidados** | **11** | **Conteo Perfecto.** El algoritmo detectó exactamente a las 11 vacas físicas, sin fragmentar identidades. |
| **Promedio Detecciones/Frame** | 10.29 | YOLO ve un promedio de 10 vacas simultáneas de forma constante en cada fotograma. |
| **ID Switches** | **0** | **Estabilidad Perfecta.** Ningún número saltó o se cruzó con otra vaca. |
| **Duración Promedio (Tracks)** | 464.18 frames | El tracker no "olvida" a las vacas gracias al anclaje estático. |

---

## 🔄 Arquitectura del Pipeline (End-to-End)

```mermaid
1. Video Raw (Frame)
      ↓
2. YOLOv8m (imgsz=1280) → Genera Bounding Boxes HD
      ↓
3. Filtros Geométricos: 
   - NMS Relajado (IoU 0.70) para oclusión.
   - Anti-Mega-Cajas (IoMin > 0.80).
      ↓
4. ResNet18 Encoder → Extrae Embedding L2-Norm (256-dim)
      ↓
5. Matriz de Costos Híbrida:
   - Costo Espacial (Euclidiana de Centroides).
   - Costo Visual (Similitud Coseno).
      ↓
6. Asignación Lineal (Hungarian Algorithm)
      ↓
7. Suavizado Visual: EMA en Coordenadas + Sticky Boxes
      ↓
8. Video Renderizado + Reporte Operativo (.txt)
```

---

## 🛠️ Tecnologías

* **Visión y Detección:** YOLOv8 Medium (`ultralytics`), OpenCV.
* **Deep Learning (Re-ID):** PyTorch 2.x, Torchvision (ResNet18 pre-entrenada).
* **Búsqueda Vectorial:** FAISS (Facebook AI Similarity Search).
* **Matemática y Optimización:** NumPy, SciPy (`linear_sum_assignment`).
* **Hardware Target:** Optimizado temporalmente para CPU en entorno macOS.

## 📝 Próximos Pasos (Roadmap)
1. **Migración a GPU/CUDA:** Acelerar la inferencia de YOLOv8m y ResNet18 para lograr procesamiento en tiempo real (30 FPS).
2. **Adaptación a Ganado en Movimiento Rápido:** Calibrar el hiperparámetro `jump_dist` (actualmente límite de 100px) para trotes o corridas largas.
3. **Validación en Nuevos Entornos:** Probar el algoritmo con iluminación extrema y drones a mayor altitud.

***

**Equipo:** Emiliano Orlando  
**Estado:** MVP Completado.