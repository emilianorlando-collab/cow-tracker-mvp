# Reportes del mockup CowTrack

Esta carpeta conserva los resultados exportables generados desde la interfaz
web local.

## Estructura

- `final/`: último resultado validado después de corregir los conflictos de
  color entre Marta, Maria y Margarita.
- `archive/`: reportes técnicos y evidencias de corridas previas, incluidas
  pruebas parciales y fallidas útiles para trazabilidad.
- `historial_publico.json`: índice sanitizado de las corridas, sin rutas locales
  privadas, credenciales ni información de sesión.

## Exclusiones intencionales

No se versionan videos renderizados, uploads de usuarios, cachés `.pkl`, tokens
de Telegram ni datasets. Esos archivos permanecen en el almacenamiento local y
están cubiertos por `.gitignore`.
