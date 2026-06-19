#!/usr/bin/env python3
"""Build a per-user Re-ID gallery from the identities stored by the web app."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base04 = load_module("base04_user_gallery", SCRIPT_DIR / "04_crear_galeria_erondina.py")


def catalog_images(catalog_dir: Path):
    paths: list[str] = []
    labels: list[str] = []
    for identity_dir in sorted(path for path in catalog_dir.iterdir() if path.is_dir()):
        images = sorted(
            path
            for path in identity_dir.iterdir()
            if not path.name.startswith(".") and path.suffix.lower() in base04.IMAGE_EXTS
        )
        references = [path for path in images if not path.stem.lower().startswith("cover")]
        for image_path in references or images:
            paths.append(str(image_path.resolve()))
            labels.append(identity_dir.name)
    return base04.ImageSet(paths=paths, labels=labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crear galeria Re-ID de un usuario CowTrack")
    parser.add_argument("--catalog_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_gallery", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report_out", required=True)
    args = parser.parse_args()

    catalog_dir = Path(args.catalog_dir)
    identities = sorted(path.name for path in catalog_dir.iterdir() if path.is_dir())
    if not identities:
        raise RuntimeError("El catalogo no contiene identidades.")

    vectors: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []

    base_path = Path(args.base_gallery)
    if base_path.exists():
        data = np.load(base_path, allow_pickle=True)
        for vector, raw_label, raw_path in zip(
            data["gallery_vectors"], data["gallery_labels"], data["gallery_paths"]
        ):
            label = str(raw_label)
            if label in identities:
                vectors.append(np.asarray(vector, dtype=np.float32))
                labels.append(label)
                paths.append(str(raw_path))

    image_set = catalog_images(catalog_dir)
    if image_set.paths:
        device = base04.torch.device("cpu")
        model = base04.cargar_modelo(args.model_path, device)
        transform = base04.transforms.Compose(
            [
                base04.transforms.Resize((224, 224)),
                base04.transforms.ToTensor(),
                base04.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        new_vectors, new_labels, new_paths = base04.extraer_embeddings(
            image_set, model, device, transform
        )
        vectors.extend(new_vectors)
        labels.extend(str(label) for label in new_labels)
        paths.extend(str(path) for path in new_paths)

    if not vectors:
        raise RuntimeError("No se pudieron generar embeddings para el catalogo.")

    matrix = np.vstack(vectors).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    label_array = np.asarray(labels)
    path_array = np.asarray(paths)
    proto_vectors, proto_labels, counts = base04.calcular_prototipos(matrix, label_array)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        gallery_vectors=matrix,
        gallery_labels=label_array,
        gallery_paths=path_array,
        proto_vectors=proto_vectors,
        proto_labels=proto_labels,
    )
    report = {
        "catalog_dir": str(catalog_dir),
        "output": str(output),
        "identities": identities,
        "embedding_count": int(len(matrix)),
        "counts_by_identity": counts,
    }
    Path(args.report_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
