# Índice de scripts

Este documento organiza los scripts del proyecto según su función dentro del
MVP. La intención es preservar las pruebas técnicas realizadas durante la
investigación sin confundirlas con el pipeline final aprobado.

## Pipeline base

| Script | Rol |
| --- | --- |
| `scripts/01_entrenar_reid.py` | Entrena el extractor Re-ID general con dataset de vacas. |
| `scripts/02_evaluar_faiss.py` | Evalúa embeddings mediante FAISS y métricas de recuperación. |
| `scripts/03_inferencia_video.py` | Ejecuta la inferencia inicial de video con detección, tracking y render. |

## Pruebas intermedias Erondina

| Script | Rol dentro de la investigación |
| --- | --- |
| `scripts/04_crear_galeria_erondina.py` | Construye la primera galería de embeddings para Marta, Maria y Margarita. |
| `scripts/05_inferencia_video_erondina.py` | Primera inferencia local sobre el video Erondina usando la galería propia. |
| `scripts/06_diagnosticar_detecciones_erondina.py` | Diagnostica umbrales de detección antes de procesar el video completo. |
| `scripts/07_inferencia_video_erondina_estable.py` | Prueba de estabilización de identidades conocidas y tracks confirmados. |
| `scripts/08_reid_global_tracklets_erondina.py` | Evalúa asignación de identidad por tracklets completos en lugar de frame a frame. |
| `scripts/09_reid_global_color_erondina.py` | Explora fusión de embeddings con evidencia de color y conteo esperado. |
| `scripts/10_reid_global_color_erondina_sin_solapes.py` | Corrige fusiones inválidas evitando unir trayectorias con solape temporal. |
| `scripts/11_reid_global_embeddings_erondina.py` | Prueba Re-ID global basada solamente en embeddings. |
| `scripts/12_finetune_reid_erondina.py` | Explora fine-tuning específico con imágenes catalogadas de Erondina. |
| `scripts/13_reid_global_embeddings_auto_erondina.py` | Automatiza la asignación global de identidades sin mapeos manuales. |
| `scripts/14_filtrar_galeria_embeddings_erondina.py` | Filtra imágenes ambiguas de la galería por consistencia de embeddings. |
| `scripts/15_crear_galeria_enfocada_erondina.py` | Genera la galería final enfocada, reduciendo ruido de bordes y vacas vecinas. |

## Pipeline final aprobado

| Script | Rol |
| --- | --- |
| `scripts/16_reid_timeline_erondina.py` | Pipeline final de análisis offline, auditoría de continuidad y render HD para Erondina. |

## Criterio profesional de versionado

Los scripts intermedios se versionan porque documentan decisiones técnicas
relevantes: diagnóstico de detecciones, intentos con color, validación sin
solapes, embeddings globales, fine-tuning y filtrado de galería. Esto permite
reproducir el camino experimental que llevó al resultado final.

Los artefactos pesados generados por estos scripts no se versionan:

- Videos `.mp4`
- Caches `.pkl`
- Datasets
- Zips
- Resultados dentro de `datos/`

El estado final del MVP se debe evaluar con `scripts/16_reid_timeline_erondina.py`
y los reportes ubicados en `reports/final/`.
