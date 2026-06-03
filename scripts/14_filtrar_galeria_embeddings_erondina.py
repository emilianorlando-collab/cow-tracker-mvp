#!/usr/bin/env python3
"""
Script 14: Filtrado automatico de galeria Erondina por consistencia.

No usa reglas de color ni mapeos manuales. Parte del archivo de embeddings
creado desde las carpetas de Erondina y elimina imagenes de galeria ambiguas:
una imagen se considera ambigua cuando su embedding se parece tanto o mas a
otra identidad que a su propia identidad.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    return mat.astype(np.float32) / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)


def calcular_prototipos(vectors: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    by_label: Dict[str, List[np.ndarray]] = defaultdict(list)
    for vector, label in zip(vectors, labels):
        by_label[str(label)].append(vector)

    proto_vectors = []
    proto_labels = []
    counts = {}
    for label in sorted(by_label):
        mat = np.vstack(by_label[label]).astype(np.float32)
        proto = np.mean(mat, axis=0, dtype=np.float32)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        proto_vectors.append(proto)
        proto_labels.append(label)
        counts[label] = len(by_label[label])
    return np.vstack(proto_vectors).astype(np.float32), np.array(proto_labels), counts


def auditar_galeria(vectors: np.ndarray, labels: np.ndarray, paths: np.ndarray):
    sims = vectors @ vectors.T
    rows = []
    for i, label in enumerate(labels):
        label = str(label)
        same = [j for j, other in enumerate(labels) if str(other) == label and j != i]
        other = [j for j, other_label in enumerate(labels) if str(other_label) != label]
        own_score = float(np.max(sims[i, same])) if same else 0.0
        if other:
            other_scores = sims[i, other]
            best_other_pos = int(np.argmax(other_scores))
            best_other_idx = int(other[best_other_pos])
            other_score = float(sims[i, best_other_idx])
            other_label = str(labels[best_other_idx])
        else:
            best_other_idx = -1
            other_score = 0.0
            other_label = ""
        rows.append(
            {
                "index": int(i),
                "label": label,
                "path": str(paths[i]),
                "nearest_own_score": own_score,
                "nearest_other_label": other_label,
                "nearest_other_index": best_other_idx,
                "nearest_other_score": other_score,
                "margin": float(own_score - other_score),
            }
        )
    return rows


def seleccionar_indices(rows, min_margin: float, min_own_score: float, min_keep_per_label: int):
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    keep = set()
    for label, label_rows in by_label.items():
        clean = [
            row for row in label_rows
            if row["margin"] >= min_margin and row["nearest_own_score"] >= min_own_score
        ]
        for row in clean:
            keep.add(row["index"])

        if len(clean) < min_keep_per_label:
            ranked = sorted(
                label_rows,
                key=lambda row: (row["margin"], row["nearest_own_score"]),
                reverse=True,
            )
            for row in ranked[:min_keep_per_label]:
                keep.add(row["index"])

    return keep


def crear_hoja_visual(rows, keep, repo_root: Path, out_path: str):
    cells = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_absolute():
            path = repo_root / path
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (220, 146), (30, 30, 30))
        img = ImageOps.contain(img, (220, 136))
        cell = Image.new("RGB", (220, 180), (18, 18, 18))
        cell.paste(img, ((220 - img.width) // 2, 34 + (136 - img.height) // 2))
        draw = ImageDraw.Draw(cell)
        is_keep = row["index"] in keep
        color = (0, 140, 0) if is_keep else (150, 0, 0)
        draw.rectangle((0, 0, 220, 34), fill=color)
        status = "KEEP" if is_keep else "DROP"
        text = f"{status} {row['label']} #{row['index']} m={row['margin']:.2f}"
        draw.text((5, 5), text, fill=(255, 255, 255))
        cells.append(cell)

    if not cells:
        return None

    cols = 5
    rows_n = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 220, rows_n * 180), "white")
    for i, cell in enumerate(cells):
        sheet.paste(cell, ((i % cols) * 220, (i // cols) * 180))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path, quality=92)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Filtrar galeria Erondina por consistencia de embeddings")
    parser.add_argument("--input", type=str, default="models/erondina_gallery_embeddings.npz")
    parser.add_argument("--output", type=str, default="models/erondina_gallery_embeddings_filtrada.npz")
    parser.add_argument("--report_out", type=str, default="reports/14_galeria_filtrada_erondina.json")
    parser.add_argument("--sheet_out", type=str, default="reports/14_galeria_filtrada_erondina.jpg")
    parser.add_argument("--min_margin", type=float, default=0.03)
    parser.add_argument("--min_own_score", type=float, default=0.75)
    parser.add_argument("--min_keep_per_label", type=int, default=8)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    vectors = normalize_rows(data["gallery_vectors"].astype(np.float32))
    labels = np.array([str(x) for x in data["gallery_labels"]])
    paths = np.array([str(x) for x in data["gallery_paths"]])

    rows = auditar_galeria(vectors, labels, paths)
    keep = seleccionar_indices(rows, args.min_margin, args.min_own_score, args.min_keep_per_label)
    keep_idxs = np.array(sorted(keep), dtype=np.int64)

    filtered_vectors = vectors[keep_idxs]
    filtered_labels = labels[keep_idxs]
    filtered_paths = paths[keep_idxs]
    proto_vectors, proto_labels, counts = calcular_prototipos(filtered_vectors, filtered_labels)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        gallery_vectors=filtered_vectors.astype(np.float32),
        gallery_labels=filtered_labels,
        gallery_paths=filtered_paths,
        proto_vectors=proto_vectors,
        proto_labels=proto_labels,
    )

    repo_root = Path.cwd()
    sheet = crear_hoja_visual(rows, keep, repo_root, args.sheet_out)
    removed = [row for row in rows if row["index"] not in keep]
    kept = [row for row in rows if row["index"] in keep]
    report = {
        "input": args.input,
        "output": args.output,
        "sheet_out": sheet,
        "original_count": int(len(labels)),
        "filtered_count": int(len(filtered_labels)),
        "removed_count": int(len(removed)),
        "counts_by_identity": counts,
        "params": vars(args),
        "kept": kept,
        "removed": removed,
        "note": (
            "Filtro automatico por consistencia de embeddings. "
            "No usa color, nombres manuales ni informacion del video."
        ),
    }
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("============================================================")
    print("Galeria Erondina filtrada por consistencia")
    print("============================================================")
    print(f"Imagenes originales : {len(labels)}")
    print(f"Imagenes conservadas: {len(filtered_labels)}")
    print(f"Imagenes removidas  : {len(removed)}")
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")
    print(f"Archivo NPZ         : {args.output}")
    print(f"Reporte             : {args.report_out}")
    if sheet:
        print(f"Hoja visual         : {sheet}")


if __name__ == "__main__":
    main()
