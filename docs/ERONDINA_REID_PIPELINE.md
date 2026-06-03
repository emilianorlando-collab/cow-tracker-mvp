# Pipeline final Re-ID Erondina

Este documento resume el pipeline final usado para el MVP de reidentificacion
de vacas en el video de campo Erondina.

## Objetivo

Generar un video renderizado donde tres vacas catalogadas queden identificadas
con nombre estable:

- `Marta`: vaca castana.
- `Maria`: vaca negra.
- `Margarita`: vaca castana.

Las etiquetas de estas tres vacas tienen prioridad visual sobre las vacas
desconocidas y deben mantenerse mientras la vaca siga dentro del plano.

## Script principal

El pipeline final esta implementado en:

```bash
scripts/16_reid_timeline_erondina.py
```

El script preserva los pasos previos y agrega una etapa offline de linea
temporal por identidad:

1. Deteccion de vacas con YOLO + BoT-SORT.
2. Extraccion de embeddings Re-ID de cuerpo enfocado.
3. Extraccion auxiliar de embeddings de zona superior/cabeza aproximada.
4. Agrupacion de fragmentos de tracking en identidades globales.
5. Asignacion automatica de `Marta`, `Maria` y `Margarita` con la galeria
   Erondina.
6. Auditoria de continuidad de etiquetas bloqueadas.
7. Render final con prioridad visual para las vacas reidentificadas.

## Galeria de identidades Erondina

La galeria final versionada es:

```bash
models/erondina_gallery_embeddings_enfocada_filtrada.npz
```

Contiene embeddings de fotos seleccionadas manualmente y extraidas del campo
Erondina. El archivo incluye:

- `gallery_vectors`
- `gallery_labels`
- `gallery_paths`
- `proto_vectors`
- `proto_labels`

## Comando de analisis final

El analisis final se ejecuta primero para consolidar evidencia temporal,
auditar identidades y generar los artefactos de revision antes del render.

```bash
python scripts/16_reid_timeline_erondina.py \
  --process_width 1920 \
  --process_height 1080 \
  --report_out reports/16_erondina_timeline_analysis.json \
  --contact_sheet_out reports/16_erondina_timeline_contact_sheet.jpg \
  --candidate_sheet_out reports/16_erondina_timeline_candidate_sheet.jpg \
  --video_out datos/Resultados/resultado_erondina_reid_final.mp4 \
  --evidence_cache_out reports/16_erondina_timeline_evidence.pkl
```

## Comando de render final

Una vez validada la auditoria, el render se genero desde cache:

```bash
python scripts/16_reid_timeline_erondina.py \
  --process_width 1920 \
  --process_height 1080 \
  --evidence_cache_in reports/16_erondina_timeline_evidence.pkl \
  --report_out reports/16_erondina_final_render.json \
  --contact_sheet_out reports/16_erondina_final_contact_sheet.jpg \
  --candidate_sheet_out reports/16_erondina_final_candidate_sheet.jpg \
  --video_out datos/Resultados/resultado_erondina_reid_final.mp4 \
  --render
```

La version final de presentacion se consolido como:

```bash
VERSION_FINAL.mp4
```

## Metricas finales

Reporte final versionado:

```bash
reports/final/16_erondina_final_render.json
```

Metricas operativas principales:

| Metrica | Resultado |
| --- | ---: |
| Resolucion | 1920x1080 |
| Frames del video final | 1409 |
| Duracion del video final | 47.01 s |
| Vacas reales confirmadas | 13 |
| Etiquetas automaticas contabilizadas | 21 |
| Error absoluto de conteo | +8 vacas |
| Precision de conteo automatico | 61.90% |
| Recall de deteccion/conteo visual | 100.0% |
| Huecos largos en medio del plano | 0 |
| Margarita visible | 100.0% |
| Maria visible | 100.0% |
| Marta visible | 99.43% |

El conteo automatico se documenta como resultado parcial: el video final tiene
13 vacas reales confirmadas visualmente, mientras que las etiquetas automaticas
contabilizan 21 por fragmentacion de algunos animales no catalogados.

## Prioridad visual de etiquetas

En el render final:

- `Marta`, `Maria` y `Margarita` se dibujan al final, por encima de las vacas
  desconocidas.
- Si una caja desconocida se superpone con una conocida, la desconocida se
  omite para no tapar la etiqueta reidentificada.
- Las etiquetas conocidas usan mayor tamano y borde mas grueso.
- La continuidad espacial sostiene la etiqueta si el tracker cambia de ID
  interno, siempre que la vaca siga dentro del plano.

## Artefactos versionados

Reportes finales:

```bash
reports/final/
```

Pruebas tecnicas intermedias:

```bash
reports/intermediate/
```
