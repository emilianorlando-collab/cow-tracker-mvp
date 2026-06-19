# Aplicación web CowTrack

Prototipo funcional de una aplicación web para conteo, tracking y
reidentificación de rodeos mediante el pipeline real de CowTrack.

## Credenciales de prueba

```text
usuario: admin
clave: admin
```

## Ejecucion

```bash
python3 webapp/cowtrack_webapp.py
```

Abrir:

```text
http://127.0.0.1:7860
```

## Estructura local en T7

La aplicación guarda datos operativos fuera del repositorio, dentro de:

```text
/Volumes/T7/cow-tracker-mvp/webapp/
```

Carpetas principales:

- `user_data/`: usuarios, catalogos de vacas y fotos cargadas.
- `reports_webapp/`: reportes generados para el usuario final.
- `uploads/`: videos subidos desde la interfaz.
- `runs/`: resultados de procesamiento ejecutados desde la interfaz.

## Flujo principal

1. Entrar al sitio publico de CowTrack.
2. Iniciar sesion con `admin/admin`.
3. Revisar o modificar el catalogo de vacas.
4. Ejecutar `Reidentificar / Conteo`.
5. Ver el progreso del procesamiento en tiempo real.
6. Consultar el reporte amigable y enviarlo por Telegram.
