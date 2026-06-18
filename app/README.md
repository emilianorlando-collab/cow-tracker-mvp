# CowTrack Web

Interfaz web local para ejecutar el pipeline CowTrack sin escribir comandos.

La app permite:

- Seleccionar un video por ruta local o subir un archivo.
- Ejecutar el pipeline final de deteccion, conteo, tracking y Re-ID.
- Ver progreso y logs de ejecucion.
- Consultar metricas finales del reporte JSON.
- Guardar el video renderizado y los reportes.
- Enviar un resumen por Telegram de forma opcional.

## Ejecucion

```bash
python3 app/cowtrack_web.py
```

Luego abrir:

```text
http://127.0.0.1:7860
```

## Telegram

Para enviar notificaciones automaticas, crear un bot con BotFather y definir:

```bash
export COWTRACK_TELEGRAM_BOT_TOKEN="TOKEN_DEL_BOT"
export COWTRACK_TELEGRAM_CHAT_ID="CHAT_ID"
python3 app/cowtrack_web.py
```

Tambien se pueden cargar esos dos valores desde la interfaz antes de ejecutar
el pipeline.

## Recomendacion de uso

Para videos grandes conviene pegar la ruta local del archivo en el disco T7 en
lugar de subirlo desde el navegador. La subida esta pensada para pruebas mas
chicas o videos livianos.

Por defecto, la interfaz queda apuntando al clip base de presentacion:

```text
/Volumes/T7/cow-tracker-mvp/datos/Resultado final/archivo a procesar.mp4
```
