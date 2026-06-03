# Scripts

Esta carpeta contiene los scripts del pipeline y las pruebas tecnicas del MVP.

Orden general:

- `01_entrenar_reid.py`: entrenamiento del extractor Re-ID general.
- `02_evaluar_faiss.py`: evaluacion vectorial con FAISS.
- `03_inferencia_video.py`: inferencia inicial de deteccion, tracking y conteo.
- `04` a `15`: pruebas y refinamientos intermedios sobre Erondina.
- `16_reid_timeline_erondina.py`: pipeline final de analisis temporal, Re-ID y
  render.

El script principal del resultado final es:

```bash
python3 scripts/16_reid_timeline_erondina.py
```

Los pesos YOLO incluidos son artefactos de deteccion usados por los scripts.
Los videos, datasets y caches deben permanecer fuera de esta carpeta.
