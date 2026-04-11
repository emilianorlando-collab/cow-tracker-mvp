(.venv) emilianoorlando@MacBook-Air-de-Emiliano cow-tracker-mvp % python3 scripts/01_entrenar_reid.py
Dispositivo forzado: cpu
⏳ Escaneando carpetas de forma segura (búsqueda profunda)...
✅ Escaneo exitoso: 46340 imágenes válidas detectadas.
📊 Resumen de División (Lógica: Última carpeta a Test / Fallback 80-20):
--------------------------------------------------------------------------------
🐄 Clase '0 ' (4 subcarpetas) -> Train (3 subc.): 824 fotos | Test (Carpeta '3'): 519 fotos
🐄 Clase '1 ' (2 subcarpetas) -> Train (1 subc.): 125 fotos | Test (Carpeta '1'): 55 fotos
🐄 Clase '10' (6 subcarpetas) -> Train (5 subc.): 1168 fotos | Test (Carpeta '5'): 495 fotos
🐄 Clase '11' (2 subcarpetas) -> Train (1 subc.): 155 fotos | Test (Carpeta '3'): 159 fotos
🐄 Clase '12' (2 subcarpetas) -> Train (1 subc.): 159 fotos | Test (Carpeta '6'): 150 fotos
🐄 Clase '13' (5 subcarpetas) -> Train (4 subc.): 499 fotos | Test (Carpeta '6'): 64 fotos
🐄 Clase '14' (8 subcarpetas) -> Train (7 subc.): 462 fotos | Test (Carpeta '7'): 159 fotos
🐄 Clase '15' (13 subcarpetas) -> Train (12 subc.): 4585 fotos | Test (Carpeta '9'): 343 fotos
🐄 Clase '16' (10 subcarpetas) -> Train (9 subc.): 3580 fotos | Test (Carpeta '9'): 319 fotos
🐄 Clase '17' (6 subcarpetas) -> Train (5 subc.): 1107 fotos | Test (Carpeta '8'): 70 fotos
🐄 Clase '18' (5 subcarpetas) -> Train (4 subc.): 411 fotos | Test (Carpeta '7'): 20 fotos
🐄 Clase '19' (6 subcarpetas) -> Train (5 subc.): 1827 fotos | Test (Carpeta '6'): 295 fotos
🐄 Clase '2 ' (2 subcarpetas) -> Train (1 subc.): 519 fotos | Test (Carpeta '3'): 162 fotos
🐄 Clase '20' (11 subcarpetas) -> Train (10 subc.): 3430 fotos | Test (Carpeta '9'): 325 fotos
🐄 Clase '21' (5 subcarpetas) -> Train (4 subc.): 1656 fotos | Test (Carpeta '5'): 155 fotos
🐄 Clase '22' (3 subcarpetas) -> Train (2 subc.): 443 fotos | Test (Carpeta '2'): 445 fotos
🐄 Clase '3 ' (8 subcarpetas) -> Train (7 subc.): 2977 fotos | Test (Carpeta '9'): 157 fotos
🐄 Clase '4 ' (13 subcarpetas) -> Train (12 subc.): 4423 fotos | Test (Carpeta '9'): 343 fotos
🐄 Clase '5 ' (8 subcarpetas) -> Train (7 subc.): 1064 fotos | Test (Carpeta '7'): 85 fotos
🐄 Clase '6 ' (9 subcarpetas) -> Train (8 subc.): 3287 fotos | Test (Carpeta '8'): 319 fotos
🐄 Clase '7 ' (8 subcarpetas) -> Train (7 subc.): 1821 fotos | Test (Carpeta '7'): 495 fotos
🐄 Clase '8 ' (5 subcarpetas) -> Train (4 subc.): 1210 fotos | Test (Carpeta '4'): 159 fotos
🐄 Clase '9 ' (16 subcarpetas) -> Train (15 subc.): 5090 fotos | Test (Carpeta '9'): 225 fotos
--------------------------------------------------------------------------------
✅ Total Imágenes: 46340
✅ Subset Train: 40822
✅ Subset Test (Prueba): 5518
🚀 Iniciando entrenamiento (Validando solo con datos Train en consola)...
Época [1/10] - Loss: 0.4348 - Train Accuracy: 0.9284
Época [2/10] - Loss: 0.0332 - Train Accuracy: 0.9937
Época [3/10] - Loss: 0.0158 - Train Accuracy: 0.9966
Época [4/10] - Loss: 0.0135 - Train Accuracy: 0.9967
Época [5/10] - Loss: 0.0053 - Train Accuracy: 0.9989
Época [6/10] - Loss: 0.0004 - Train Accuracy: 1.0000
Época [7/10] - Loss: 0.0081 - Train Accuracy: 0.9975
Época [8/10] - Loss: 0.0059 - Train Accuracy: 0.9983
Época [9/10] - Loss: 0.0067 - Train Accuracy: 0.9982
Época [10/10] - Loss: 0.0043 - Train Accuracy: 0.9987
🎉 Extractor Re-ID (Libre de Data Leakage) guardado en: models/mi_modelo_reid.pt
(.venv) emilianoorlando@MacBook-Air-de-Emiliano cow-tracker-mvp % 
