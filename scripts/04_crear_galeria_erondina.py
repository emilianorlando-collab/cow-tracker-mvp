#!/usr/bin/env python3
"""
Script 04: Construcción de galería Re-ID específica de Erondina.

Usa el extractor entrenado con OpenCows (models/mi_modelo_reid.pt) para crear
embeddings de las vacas conocidas del campo. La estructura esperada es:

datos/erondina_reid/
├── Marta/
│   ├── galeria/
│   └── test/
├── María/
│   ├── galeria/
│   └── test/
└── Margarita/
    ├── galeria/
    └── test/
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


class ReIDFeatureExtractor(nn.Module):
    """Extractor de embeddings 256-D (ResNet18 backbone + Linear 512->256 + L2)."""

    def __init__(self):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.embedding = nn.Linear(512, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x


@dataclass
class ImageSet:
    paths: List[str]
    labels: List[str]


def cargar_modelo(model_path: str, device: torch.device) -> nn.Module:
    model = ReIDFeatureExtractor().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def listar_imagenes(root_dir: str, split: str) -> ImageSet:
    paths: List[str] = []
    labels: List[str] = []

    for label in sorted(os.listdir(root_dir)):
        cow_dir = os.path.join(root_dir, label)
        split_dir = os.path.join(cow_dir, split)
        if not os.path.isdir(split_dir):
            continue

        for current_dir, _, files in os.walk(split_dir):
            for file_name in sorted(files):
                if file_name.startswith("."):
                    continue
                if not file_name.lower().endswith(IMAGE_EXTS):
                    continue
                paths.append(os.path.join(current_dir, file_name))
                labels.append(label)

    return ImageSet(paths=paths, labels=labels)


def extraer_embeddings(
    image_set: ImageSet,
    model: nn.Module,
    device: torch.device,
    tfm: transforms.Compose,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors: List[np.ndarray] = []
    labels_ok: List[str] = []
    paths_ok: List[str] = []

    with torch.no_grad():
        for path, label in zip(image_set.paths, image_set.labels):
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    x = tfm(img).unsqueeze(0).to(device)
                emb = model(x).cpu().numpy()[0].astype(np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                vectors.append(emb)
                labels_ok.append(label)
                paths_ok.append(path)
            except Exception as exc:
                print(f"⚠️  No se pudo procesar {path}: {exc}")

    if not vectors:
        raise RuntimeError("No se pudo extraer ningún embedding válido.")

    return (
        np.vstack(vectors).astype(np.float32),
        np.array(labels_ok),
        np.array(paths_ok),
    )


def calcular_prototipos(vectors: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    by_label: Dict[str, List[np.ndarray]] = defaultdict(list)
    for vector, label in zip(vectors, labels):
        by_label[str(label)].append(vector)

    proto_vectors: List[np.ndarray] = []
    proto_labels: List[str] = []
    counts: Dict[str, int] = {}

    for label in sorted(by_label.keys()):
        mat = np.vstack(by_label[label]).astype(np.float32)
        proto = np.mean(mat, axis=0, dtype=np.float32)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        proto_vectors.append(proto)
        proto_labels.append(label)
        counts[label] = len(by_label[label])

    return np.vstack(proto_vectors).astype(np.float32), np.array(proto_labels), counts


def evaluar_test(
    test_vectors: np.ndarray,
    test_labels: np.ndarray,
    proto_vectors: np.ndarray,
    proto_labels: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    sims = test_vectors @ proto_vectors.T
    best_idx = np.argmax(sims, axis=1)
    best_scores = sims[np.arange(len(test_vectors)), best_idx]

    total = len(test_labels)
    correct = 0
    rejected = 0
    false_accepts = 0
    rows = []
    per_label = {
        str(label): {"tp": 0, "fp": 0, "fn": 0}
        for label in sorted(set(test_labels.tolist()) | set(proto_labels.tolist()))
    }

    for true_label, pred_idx, score in zip(test_labels, best_idx, best_scores):
        true_label = str(true_label)
        pred_label = str(proto_labels[pred_idx])
        accepted = float(score) >= threshold

        if not accepted:
            rejected += 1
            pred_out = "desconocida"
            per_label[true_label]["fn"] += 1
        else:
            pred_out = pred_label
            if pred_label == true_label:
                correct += 1
                per_label[true_label]["tp"] += 1
            else:
                false_accepts += 1
                per_label[pred_label]["fp"] += 1
                per_label[true_label]["fn"] += 1

        rows.append(
            {
                "true_label": true_label,
                "pred_label": pred_out,
                "score": float(score),
                "accepted": accepted,
            }
        )

    per_label_metrics = {}
    precisions = []
    recalls = []
    f1s = []
    for label, counts in per_label.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        per_label_metrics[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "total": total,
        "threshold": threshold,
        "accuracy_thresholded": (100.0 * correct / total) if total else 0.0,
        "precision_macro": float(np.mean(precisions)) if precisions else 0.0,
        "recall_macro": float(np.mean(recalls)) if recalls else 0.0,
        "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "rejection_rate": (100.0 * rejected / total) if total else 0.0,
        "false_accepts": false_accepts,
        "per_label": per_label_metrics,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Crear galería de embeddings para vacas de Erondina")
    parser.add_argument("--data_dir", type=str, default="datos/erondina_reid")
    parser.add_argument("--model_path", type=str, default="models/mi_modelo_reid.pt")
    parser.add_argument("--output", type=str, default="models/erondina_gallery_embeddings.npz")
    parser.add_argument("--report_out", type=str, default="reports/04_galeria_erondina.json")
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    device = torch.device("cpu")
    model = cargar_modelo(args.model_path, device)
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    gallery_set = listar_imagenes(args.data_dir, "galeria")
    test_set = listar_imagenes(args.data_dir, "test")

    print(f"Galería: {len(gallery_set.paths)} imágenes")
    print(f"Test: {len(test_set.paths)} imágenes")

    gallery_vectors, gallery_labels, gallery_paths = extraer_embeddings(gallery_set, model, device, tfm)
    proto_vectors, proto_labels, counts = calcular_prototipos(gallery_vectors, gallery_labels)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        gallery_vectors=gallery_vectors,
        gallery_labels=gallery_labels,
        gallery_paths=gallery_paths,
        proto_vectors=proto_vectors,
        proto_labels=proto_labels,
    )

    report: Dict[str, object] = {
        "data_dir": args.data_dir,
        "model_path": args.model_path,
        "output": args.output,
        "counts_by_identity": counts,
        "num_gallery_images": int(len(gallery_vectors)),
        "num_identities": int(len(proto_labels)),
    }

    if test_set.paths:
        test_vectors, test_labels, _ = extraer_embeddings(test_set, model, device, tfm)
        report["test_eval"] = evaluar_test(
            test_vectors=test_vectors,
            test_labels=test_labels,
            proto_vectors=proto_vectors,
            proto_labels=proto_labels,
            threshold=args.threshold,
        )

    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("============================================================")
    print("Galería Erondina creada")
    print("============================================================")
    for label in proto_labels:
        print(f"{label}: {counts[str(label)]} imágenes de galería")
    if "test_eval" in report:
        eval_report = report["test_eval"]
        print(f"Accuracy test con umbral {args.threshold:.2f}: {eval_report['accuracy_thresholded']:.2f}%")
        print(f"Precision macro test: {100.0 * eval_report['precision_macro']:.2f}%")
        print(f"Recall macro test: {100.0 * eval_report['recall_macro']:.2f}%")
        print(f"F1 macro test: {100.0 * eval_report['f1_macro']:.2f}%")
        print(f"Rechazos test: {eval_report['rejection_rate']:.2f}%")
        print(f"Falsas aceptaciones test: {eval_report['false_accepts']}")
    print(f"Embeddings guardados en: {args.output}")
    print(f"Reporte guardado en: {args.report_out}")


if __name__ == "__main__":
    main()
