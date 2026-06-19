#!/bin/zsh

set -u

INSTALL_ROOT="$HOME/Library/Application Support/CowTrack"
RUNTIME_ROOT="$INSTALL_ROOT/runtime"
STORAGE_ROOT="$INSTALL_ROOT/storage"

export COWTRACK_DATA_ROOT="$STORAGE_ROOT"
export COWTRACK_PYTHON="$INSTALL_ROOT/.venv/bin/python3"

cd "$RUNTIME_ROOT" || exit 1
exec /usr/bin/python3 "$RUNTIME_ROOT/webapp/cowtrack_webapp.py"
