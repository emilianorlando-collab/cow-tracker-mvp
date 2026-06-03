#!/usr/bin/env python3
"""
Script 12: Fine-tuning Re-ID especifico para Erondina.

No usa reglas manuales de color. Ajusta el extractor `mi_modelo_reid.pt` con las
identidades reales de Erondina y genera una galeria nueva de embeddings.

Entrena con:
  datos/erondina_reid/<identidad>/galeria

Evalua con:
  datos/erondina_reid/<identidad>/test
"""

import argparse
import json
import os
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def label_key(label: str) -> str:
    text = unicodedata.normalize("NFD", str(label))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.lower()


def display_label(label: str) -> str:
    mapping = {"maria": "Maria", "marta": "Marta", "margarita": "Margarita"}
    return mapping.get(label_key(label), str(label))


class ReIDFineTuneModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.embedding = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward_features(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, x):
        emb = self.forward_features(x)
        return self.classifier(emb)


class ReIDFeatureExtractor(nn.Module):
    def __init__(self, trained_model: ReIDFineTuneModel):
        super().__init__()
        self.backbone = trained_model.backbone
        self.embedding = trained_model.embedding

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)


class CowImageDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples: List[Tuple[str, str]] = []

        labels = []
        for label in sorted(os.listdir(root_dir)):
            split_dir = os.path.join(root_dir, label, split)
            if not os.path.isdir(split_dir):
                continue
            labels.append(label_key(label))
            for file_name in sorted(os.listdir(split_dir)):
                if file_name.startswith(".") or not file_name.lower().endswith(IMAGE_EXTS):
                    continue
                self.samples.append((os.path.join(split_dir, file_name), label_key(label)))

        self.classes = sorted(set(labels))
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label], path, label


def load_base_extractor(model: ReIDFineTuneModel, model_path: str, device):
    state = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Base cargada: missing={len(missing)} unexpected={len(unexpected)}")


def extract_embeddings(dataset: CowImageDataset, extractor, device, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    vectors, labels, paths = [], [], []
    extractor.eval()
    with torch.no_grad():
        for images, _, batch_paths, batch_labels in loader:
            images = images.to(device)
            emb = extractor(images).cpu().numpy().astype(np.float32)
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
            vectors.append(emb)
            labels.extend(list(batch_labels))
            paths.extend(list(batch_paths))
    return np.vstack(vectors).astype(np.float32), np.array(labels), np.array(paths)


def evaluate_all_vectors(gallery_vecs, gallery_labels, test_vecs, test_labels):
    rows = []
    labels = sorted(set(gallery_labels.tolist()))
    correct = 0
    confusion = Counter()
    for vec, true_label in zip(test_vecs, test_labels):
        scores = {}
        for label in labels:
            idxs = np.where(gallery_labels == label)[0]
            sims = gallery_vecs[idxs] @ vec
            scores[label] = float(np.max(sims))
        pred = max(scores, key=scores.get)
        correct += int(pred == true_label)
        confusion[(display_label(true_label), display_label(pred))] += 1
        rows.append(
            {
                "true_label": display_label(true_label),
                "pred_label": display_label(pred),
                "score": scores[pred],
                "all_scores": {display_label(k): v for k, v in scores.items()},
            }
        )
    total = len(test_labels)
    return {
        "accuracy": correct / total if total else 0.0,
        "total": total,
        "confusion": {f"{a}->{b}": v for (a, b), v in confusion.items()},
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning ReID Erondina")
    parser.add_argument("--data_dir", type=str, default="datos/erondina_reid")
    parser.add_argument("--base_model", type=str, default="models/mi_modelo_reid.pt")
    parser.add_argument("--output_model", type=str, default="models/erondina_reid_finetuned.pt")
    parser.add_argument("--output_gallery", type=str, default="models/erondina_gallery_embeddings_finetuned.npz")
    parser.add_argument("--report_out", type=str, default="reports/12_finetune_reid_erondina.json")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device("cpu")
    train_tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=4, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = CowImageDataset(args.data_dir, "galeria", train_tfm)
    gallery_ds = CowImageDataset(args.data_dir, "galeria", eval_tfm)
    test_ds = CowImageDataset(args.data_dir, "test", eval_tfm)
    if len(train_ds.classes) < 2:
        raise RuntimeError("Se requieren al menos 2 identidades para fine-tuning.")

    model = ReIDFineTuneModel(num_classes=len(train_ds.classes)).to(device)
    load_base_extractor(model, args.base_model, device)

    optimizer = torch.optim.Adam(
        [
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.embedding.parameters(), "lr": args.lr_head},
            {"params": model.classifier.parameters(), "lr": args.lr_head},
        ]
    )
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_ok = 0
        total_n = 0
        for images, labels, _, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_ok += int((logits.argmax(dim=1) == labels).sum().item())
            total_n += images.size(0)
        item = {
            "epoch": epoch + 1,
            "loss": total_loss / max(1, total_n),
            "train_acc": total_ok / max(1, total_n),
        }
        history.append(item)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{args.epochs} loss={item['loss']:.4f} acc={item['train_acc']:.3f}")

    extractor = ReIDFeatureExtractor(model).to(device)
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save(extractor.state_dict(), args.output_model)

    gallery_vecs, gallery_labels, gallery_paths = extract_embeddings(gallery_ds, extractor, device, args.batch_size)
    test_vecs, test_labels, test_paths = extract_embeddings(test_ds, extractor, device, args.batch_size)
    eval_report = evaluate_all_vectors(gallery_vecs, gallery_labels, test_vecs, test_labels)

    np.savez_compressed(
        args.output_gallery,
        gallery_vectors=gallery_vecs,
        gallery_labels=gallery_labels,
        gallery_paths=gallery_paths,
    )

    report = {
        "classes": [display_label(x) for x in train_ds.classes],
        "num_gallery": len(gallery_ds),
        "num_test": len(test_ds),
        "history": history,
        "eval": eval_report,
        "output_model": args.output_model,
        "output_gallery": args.output_gallery,
    }
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("============================================================")
    print("Fine-tuning Erondina completado")
    print("============================================================")
    print(f"Accuracy test all-vectors: {100.0 * eval_report['accuracy']:.2f}% ({eval_report['total']} imgs)")
    print(f"Modelo guardado en: {args.output_model}")
    print(f"Galeria guardada en: {args.output_gallery}")
    print(f"Reporte guardado en: {args.report_out}")


if __name__ == "__main__":
    main()
