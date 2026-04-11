(.venv) emilianoorlando@MacBook-Air-de-Emiliano cow-tracker-mvp % python3 scripts/02_evaluar_faiss.py --threshold 0.85
📊 Construyendo Evaluación (Reflejando el Split del Script 01)...
✅ Galería construida: 1150 imágenes de posiciones conocidas.
✅ Queries construidos: 1120 imágenes de posiciones INÉDITAS.
Extrayendo características de 1150 imágenes...
Extrayendo características de 1120 imágenes...
======================================================================
 MVP Cow Tracker - Closed-set (CROSS-POSE STRICT)
======================================================================
Métrica              | All-vectors     | Prototype      
----------------------------------------------------------------------
Top-1 Accuracy       |         62.32% |         61.79%
Top-5 Accuracy       |         62.41% |         85.09%
Gallery vectors      |           1150 |             23
Query count          |           1120 |           1120
False accepts        |            422 |            428
======================================================================
 MVP Cow Tracker - Open-set (Threshold = 0.85)
======================================================================
Config               | Top-1 thresholded  | Rejection rate  | False accepts
----------------------------------------------------------------------
All-vectors + thr    |              51.25% |         38.21% |           118
Prototype + thr      |              50.00% |         40.09% |           111
======================================================================
