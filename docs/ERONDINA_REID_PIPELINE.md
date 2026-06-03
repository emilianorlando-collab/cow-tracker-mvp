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

Contiene embeddings de las fotos seleccionadas manualmente desde el propio
video de Erondina. El archivo incluye:

- `gallery_vectors`
- `gallery_labels`
- `gallery_paths`
- `proto_vectors`
- `proto_labels`

## Comando de analisis final

El analisis final se ejecuto sobre la segunda mitad del video original, desde
el frame `4880`, equivalente a aproximadamente `00:02:43` del video completo.

```bash
python scripts/16_reid_timeline_erondina.py \
  --process_width 1920 \
  --process_height 1080 \
  --start_frame 4880 \
  --report_out reports/16_segunda_mitad_original_timeline_hd_head_analysis.json \
  --contact_sheet_out reports/16_contact_sheet_segunda_mitad_original_timeline_hd_head_analysis.jpg \
  --candidate_sheet_out reports/16_candidate_sheet_segunda_mitad_original_timeline_hd_head_analysis.jpg \
  --video_out datos/Resultados/resultado_erondina_reid_timeline_hd_head_segunda_mitad_original.mp4 \
  --evidence_cache_out reports/16_evidence_segunda_mitad_original_timeline_hd_head.pkl
```

El cache `.pkl` es un artefacto pesado/intermedio y no debe subirse al repo.

## Comando de render final

Una vez validada la auditoria, el render se genero desde cache:

```bash
python scripts/16_reid_timeline_erondina.py \
  --process_width 1920 \
  --process_height 1080 \
  --start_frame 4880 \
  --evidence_cache_in reports/16_evidence_segunda_mitad_original_timeline_hd_head.pkl \
  --report_out reports/16_segunda_mitad_original_timeline_hd_head_render.json \
  --contact_sheet_out reports/16_contact_sheet_segunda_mitad_original_timeline_hd_head_render.jpg \
  --candidate_sheet_out reports/16_candidate_sheet_segunda_mitad_original_timeline_hd_head_render.jpg \
  --video_out datos/Resultados/resultado_erondina_reid_timeline_hd_head_segunda_mitad_original.mp4 \
  --render
```

Luego se recorto el video desde `00:01:08` del render de segunda mitad para la
version de presentacion:

```bash
RESULTADO_FINAL.mp4
```

Este video final queda en `datos/` o en la carpeta local de trabajo, pero no se
versiona en Git porque supera el limite practico de GitHub.

## Metricas finales

Reporte final versionado:

```bash
reports/final/16_segunda_mitad_original_timeline_hd_head_render.json
```

Metricas operativas principales:

| Metrica | Resultado |
| --- | ---: |
| Frames procesados | 4881 |
| Resolucion | 1920x1080 |
| Conteo estimado | 14 |
| Tracks globales internos | 22 |
| Accuracy conteo vs 14 | 100% |
| Huecos largos en medio del plano | 0 |
| Margarita visible | 100.0% |
| Maria visible | 99.75% |
| Marta visible | 96.13% |

El contador de tracks internos (`22`) no representa vacas reales, sino
fragmentacion causada por giros del dron, oclusiones y cambios de angulo. Para
conteo se usa la estimacion por vacas visibles por frame y fusion de
fragmentos.

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

Artefactos pesados no versionados:

- Videos `.mp4`
- Datasets y zips
- Caches `.pkl`
- Videos raw del dron
