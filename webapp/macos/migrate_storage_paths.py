#!/usr/bin/env python3
"""Rewrite absolute paths inside CowTrack JSON files after a storage import."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def replace_paths(value, source: str, destination: str):
    if isinstance(value, dict):
        return {key: replace_paths(item, source, destination) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_paths(item, source, destination) for item in value]
    if isinstance(value, str) and value.startswith(source):
        return destination + value[len(source) :]
    return value


def main() -> None:
    storage_root = Path(sys.argv[1])
    source = sys.argv[2].rstrip("/")
    destination = sys.argv[3].rstrip("/")
    for json_path in storage_root.rglob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        migrated = replace_paths(data, source, destination)
        if migrated != data:
            json_path.write_text(
                json.dumps(migrated, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
