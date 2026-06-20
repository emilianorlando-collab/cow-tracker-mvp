#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
SERVICE_LABEL="com.cowtrack.webapp"
DESTINATION="$HOME/Library/LaunchAgents/$SERVICE_LABEL.plist"
INSTALL_ROOT="$HOME/Library/Application Support/CowTrack"
RUNTIME_DIR="$INSTALL_ROOT/runtime"
STORAGE_DIR="$INSTALL_ROOT/storage"
T7_ROOT="/Volumes/T7/cow-tracker-mvp"
DOMAIN="gui/$(id -u)"

mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$INSTALL_ROOT" "$RUNTIME_DIR/scripts" "$RUNTIME_DIR/models" "$RUNTIME_DIR/app/runs" "$STORAGE_DIR"

ditto "$REPO_ROOT/webapp" "$RUNTIME_DIR/webapp"
for script in 04_crear_galeria_erondina.py 05_inferencia_video_erondina.py 13_reid_global_embeddings_auto_erondina.py 16_reid_timeline_erondina.py 17_crear_galeria_usuario.py yolov8m.pt; do
  cp "$REPO_ROOT/scripts/$script" "$RUNTIME_DIR/scripts/$script"
done
cp "$REPO_ROOT/models/mi_modelo_reid.pt" "$RUNTIME_DIR/models/mi_modelo_reid.pt"
cp "$REPO_ROOT/models/erondina_gallery_embeddings_enfocada_filtrada.npz" "$RUNTIME_DIR/models/erondina_gallery_embeddings_enfocada_filtrada.npz"
if [[ -d "$T7_ROOT/datos/erondina_reid" ]]; then
  mkdir -p "$RUNTIME_DIR/datos"
  ditto "$T7_ROOT/datos/erondina_reid" "$RUNTIME_DIR/datos/erondina_reid"
fi

if [[ -d "$REPO_ROOT/app/runs/20260617_201441_cad968e6" ]]; then
  ditto "$REPO_ROOT/app/runs/20260617_201441_cad968e6" "$RUNTIME_DIR/app/runs/20260617_201441_cad968e6"
fi
if [[ ! -d "$INSTALL_ROOT/.venv" ]]; then
  [[ -d "$T7_ROOT/.venv" ]] || { echo "No se encontró el entorno Python en el T7." >&2; exit 1; }
  ditto "$T7_ROOT/.venv" "$INSTALL_ROOT/.venv"
fi
if [[ ! -f "$STORAGE_DIR/.original_data_imported" ]]; then
  mkdir -p "$STORAGE_DIR/webapp" "$STORAGE_DIR/datos/Resultado final"
  for directory in user_data reports_webapp runs uploads contactos soporte; do
    if [[ -d "$T7_ROOT/webapp/$directory" ]]; then
      ditto "$T7_ROOT/webapp/$directory" "$STORAGE_DIR/webapp/$directory"
    fi
  done
  for directory in recursos_web fotos_de_perfil; do
    if [[ -d "$T7_ROOT/datos/$directory" ]]; then
      ditto "$T7_ROOT/datos/$directory" "$STORAGE_DIR/datos/$directory"
    fi
  done
  for file in "Publicidad 3.mp4"; do
    [[ -f "$T7_ROOT/datos/$file" ]] && cp "$T7_ROOT/datos/$file" "$STORAGE_DIR/datos/$file"
  done
  if [[ -f "$T7_ROOT/datos/Resultado final/RESULTADO_COWTRACK.mp4" ]]; then
    cp "$T7_ROOT/datos/Resultado final/RESULTADO_COWTRACK.mp4" "$STORAGE_DIR/datos/Resultado final/RESULTADO_COWTRACK.mp4"
  fi
  /usr/bin/python3 "$SCRIPT_DIR/migrate_storage_paths.py" "$STORAGE_DIR" "$T7_ROOT" "$STORAGE_DIR"
  touch "$STORAGE_DIR/.original_data_imported"
fi

cp "$SCRIPT_DIR/start_cowtrack.sh" "$INSTALL_ROOT/start_cowtrack.sh"
chmod +x "$INSTALL_ROOT/start_cowtrack.sh"
cp "$SCRIPT_DIR/$SERVICE_LABEL.plist" "$DESTINATION"

launchctl bootout "$DOMAIN/$SERVICE_LABEL" 2>/dev/null || true
for _ in {1..20}; do
  launchctl print "$DOMAIN/$SERVICE_LABEL" >/dev/null 2>&1 || break
  sleep 0.25
done
rm -f /tmp/cowtrack-webapp.log /tmp/cowtrack-webapp-error.log
launchctl bootstrap "$DOMAIN" "$DESTINATION"
launchctl enable "$DOMAIN/$SERVICE_LABEL"
launchctl kickstart -k "$DOMAIN/$SERVICE_LABEL"

echo "CowTrack instalado como servicio de macOS."
echo "Abrir: http://127.0.0.1:7860"
echo "Estado: launchctl print $DOMAIN/$SERVICE_LABEL"
