#!/usr/bin/env python3
"""
Script 15: Galeria Erondina con embeddings enfocados.

Usa el mismo extractor `mi_modelo_reid.pt`, pero antes del embedding aplica un
recorte central al crop/imagen. La idea es reducir informacion de vacas vecinas,
bordes y oclusiones, sin usar reglas de color ni mapeos manuales.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def cargar_modulo(nombre: str, path: Path):
    spec = importlib.util.spec_from_file_location(nombre, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base04 = cargar_modulo("base04", SCRIPT_DIR / "04_crear_galeria_erondina.py")


def focus_pil_image(img, margin_x: float, margin_y: float):
    w, h = img.size
    x1 = int(round(w * margin_x))
    y1 = int(round(h * margin_y))
    x2 = int(round(w * (1.0 - margin_x)))
    y2 = int(round(h * (1.0 - margin_y)))
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def listar_imagenes(data_dir: str, splits):
    paths = []
    labels = []
    root = Path(data_dir)
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for split in splits:
            split_dir = label_dir / split
            if not split_dir.is_dir():
                continue
            for current_dir, _, files in os.walk(split_dir):
                for file_name in sorted(files):
                    if file_name.startswith("."):
                        continue
                    if not file_name.lower().endswith(base04.IMAGE_EXTS):
                        continue
                    paths.append(str(Path(current_dir) / file_name))
                    labels.append(label_dir.name)
    return base04.ImageSet(paths=paths, labels=labels)


def extraer_embeddings_enfocados(image_set, model, device, tfm, margin_x, margin_y):
    vectors = []
    labels_ok = []
    paths_ok = []
    with base04.torch.no_grad():
        for path, label in zip(image_set.paths, image_set.labels):
            try:
                with base04.Image.open(path) as img:
                    img = img.convert("RGB")
                    img = focus_pil_image(img, margin_x, margin_y)
                    x = tfm(img).unsqueeze(0).to(device)
                emb = model(x).cpu().numpy()[0].astype(np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                vectors.append(emb)
                labels_ok.append(label)
                paths_ok.append(path)
            except Exception as exc:
                print(f"No se pudo procesar {path}: {exc}")
    if not vectors:
        raise RuntimeError("No se pudo extraer ningun embedding.")
    return np.vstack(vectors).astype(np.float32), np.array(labels_ok), np.array(paths_ok)


def main():
    parser = argparse.ArgumentParser(description="Crear galeria Erondina enfocada")
    parser.add_argument("--data_dir", type=str, default="datos/erondina_reid")
    parser.add_argument("--model_path", type=str, default="models/mi_modelo_reid.pt")
    parser.add_argument("--output", type=str, default="models/erondina_gallery_embeddings_enfocada.npz")
    parser.add_argument("--report_out", type=str, default="reports/15_galeria_enfocada_erondina.json")
    parser.add_argument("--splits", type=str, default="galeria,test")
    parser.add_argument("--focus_margin_x", type=float, default=0.12)
    parser.add_argument("--focus_margin_y", type=float, default=0.18)
    args = parser.parse_args()

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    device = base04.torch.device("cpu")
    model = base04.cargar_modelo(args.model_path, device)
    tfm = base04.transforms.Compose(
        [
            base04.transforms.Resize((224, 224)),
            base04.transforms.ToTensor(),
            base04.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_set = listar_imagenes(args.data_dir, splits)
    vectors, labels, paths = extraer_embeddings_enfocados(
        image_set,
        model,
        device,
        tfm,
        args.focus_margin_x,
        args.focus_margin_y,
    )
    proto_vectors, proto_labels, counts = base04.calcular_prototipos(vectors, labels)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        gallery_vectors=vectors,
        gallery_labels=labels,
        gallery_paths=paths,
        proto_vectors=proto_vectors,
        proto_labels=proto_labels,
    )

    report = {
        "data_dir": args.data_dir,
        "model_path": args.model_path,
        "output": args.output,
        "splits": splits,
        "focus_margin_x": args.focus_margin_x,
        "focus_margin_y": args.focus_margin_y,
        "num_images": int(len(labels)),
        "counts_by_identity": counts,
        "note": (
            "Embeddings con recorte central previo al extractor ReID. "
            "No usa color ni IDs manuales."
        ),
    }
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("============================================================")
    print("Galeria Erondina enfocada")
    print("============================================================")
    print(f"Imagenes: {len(labels)}")
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")
    print(f"NPZ    : {args.output}")
    print(f"Reporte: {args.report_out}")


if __name__ == "__main__":
    main()
