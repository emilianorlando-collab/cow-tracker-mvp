# Modelos y Embeddings

Esta carpeta contiene los artefactos de modelo necesarios para reproducir el
pipeline del MVP.

Archivos principales:

- `mi_modelo_reid.pt`: extractor Re-ID general entrenado a partir de OpenCows.
- `erondina_gallery_embeddings_enfocada_filtrada.npz`: galeria final de
  embeddings para Marta, Maria y Margarita.
- `checkpoint_epoch_*.pt`: checkpoints historicos del entrenamiento Re-ID.

La logica del proyecto separa dos niveles:

- Modelo general: aprende una representacion visual de vacas.
- Galeria Erondina: define las identidades concretas a reidentificar.

No se recomienda agregar datasets, videos ni caches en esta carpeta.
