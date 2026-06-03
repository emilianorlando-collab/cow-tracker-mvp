# Artefactos externos

Los videos, datasets y caches de ejecucion no se versionan en Git porque son
archivos pesados. La carpeta `datos/` esta ignorada por `.gitignore` y debe
usarse para:

- Videos raw del campo.
- Datasets.
- Videos renderizados.
- Resultados finales de presentacion.
- Caches `.pkl` de evidencia.

## Video final de presentacion

Archivo local generado:

```bash
/Volumes/T7/cow-tracker-mvp/RESULTADO_FINAL.mp4
```

Tambien puede ubicarse dentro de:

```bash
datos/Resultado final/RESULTADO_FINAL.mp4
```

Detalles verificados:

- Duracion: `94.86 s`
- Resolucion: `1920x1080`
- FPS: `29.97`
- Frames: `2843`
- Tamano aproximado: `571 MB`

Este archivo supera el limite practico de GitHub para archivos versionados. Si
se necesita distribuirlo desde GitHub, la opcion recomendada es subirlo como
Release Asset, no como archivo dentro del repositorio.
