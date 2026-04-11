#!/usr/bin/env python3
"""
02_evaluar_faiss.py (Versión Cross-Pose Estricta)
Alineado 100% con el Script 01.
Galería = Subcarpetas de Entrenamiento (poses conocidas).
Queries = Última subcarpeta de Test (pose inédita).
"""

# --- ESCUDO ANTI SEGMENTATION FAULT PARA MAC ---
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss
faiss.omp_set_num_threads(1) 
# -----------------------------------------------

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

# ==========================================
# 1. ARQUITECTURA
# ==========================================
class ReIDFeatureExtractor(nn.Module):
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

# ==========================================
# 2. RECOLECCIÓN CROSS-POSE (Espejo del Script 01)
# ==========================================
def recolectar_cross_pose(root_dir: str):
    """
    Agrupa las fotos exactamente como el Script 1:
    Dict[Clase] -> Dict[Subcarpeta] -> List[Rutas]
    """
    agrupado = defaultdict(lambda: defaultdict(list))
    
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"No existe: {root_dir}")
        
    carpetas = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    classes = sorted(carpetas)
    
    for class_name in classes:
        class_path = os.path.join(root_dir, class_name)
        try:
            for carpeta_actual, subcarpetas, archivos in os.walk(class_path):
                subcarpeta_name = os.path.relpath(carpeta_actual, class_path)
                for file_name in archivos:
                    if not file_name.startswith('.'):
                        if '.' not in file_name or file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            full_path = os.path.join(carpeta_actual, file_name)
                            agrupado[class_name][subcarpeta_name].append(full_path)
        except OSError:
            continue
            
    return agrupado

def construir_splits_espejo(agrupado, max_por_vaca=50) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Respeta la lógica del Script 1:
    - >1 Subcarpeta: Train subfolders -> Galería | Test subfolder -> Queries
    - 1 Subcarpeta: 80% interno -> Galería | 20% interno -> Queries
    Limitamos a 'max_por_vaca' para que tu Mac no colapse en memoria al evaluar.
    """
    gallery_paths, gallery_labels = [], []
    query_paths, query_labels = [], []

    print("\n📊 Construyendo Evaluación (Reflejando el Split del Script 01)...")
    for clase in sorted(agrupado.keys()):
        subfolders = sorted(agrupado[clase].keys())
        num_subfolders = len(subfolders)

        if num_subfolders == 1:
            sf = subfolders[0]
            fotos = agrupado[clase][sf]
            
            # Limitar a max_por_vaca para eficiencia
            fotos = fotos[:max_por_vaca]
            
            n_total = len(fotos)
            n_train = int(n_total * 0.8)

            g_paths = fotos[:n_train]
            q_paths = fotos[n_train:]
        else:
            sf_train = subfolders[:-1]
            sf_test = subfolders[-1]

            # Recolectar para Galería
            g_paths_todas = []
            for sf in sf_train:
                g_paths_todas.extend(agrupado[clase][sf])
                
            # Recolectar para Queries
            q_paths_todas = agrupado[clase][sf_test]

            # Limitamos para no ahogar la RAM
            g_paths = g_paths_todas[:max_por_vaca]
            q_paths = q_paths_todas[:max_por_vaca]

        if len(g_paths) == 0 or len(q_paths) == 0:
            continue

        gallery_paths.extend(g_paths)
        gallery_labels.extend([clase] * len(g_paths))
        query_paths.extend(q_paths)
        query_labels.extend([clase] * len(q_paths))

    print(f"✅ Galería construida: {len(gallery_paths)} imágenes de posiciones conocidas.")
    print(f"✅ Queries construidos: {len(query_paths)} imágenes de posiciones INÉDITAS.\n")
    return gallery_paths, gallery_labels, query_paths, query_labels


# ==========================================
# 3. EXTRACCIÓN Y FAISS (Idéntico a tu versión estable)
# ==========================================
def cargar_modelo(model_path: str, device: torch.device) -> nn.Module:
    model = ReIDFeatureExtractor().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def extraer_embeddings(paths, labels, model, device, tfm):
    vectors = []
    valid_labels = []
    print(f"Extrayendo características de {len(paths)} imágenes...")
    with torch.no_grad():
        for path, label in zip(paths, labels):
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    x = tfm(img).unsqueeze(0).to(device)
                emb = model(x).cpu().numpy().astype(np.float32)
                vectors.append(emb[0])
                valid_labels.append(label)
            except Exception:
                continue

    mat = np.vstack(vectors).astype(np.float32)
    faiss.normalize_L2(mat)
    return mat, valid_labels

def calcular_prototipos(gallery_vecs, gallery_labels):
    por_clase_vecs = defaultdict(list)
    for v, y in zip(gallery_vecs, gallery_labels):
        por_clase_vecs[y].append(v)
    proto_vecs, proto_labels = [], []
    for clase in sorted(por_clase_vecs.keys()):
        m = np.mean(np.vstack(por_clase_vecs[clase]), axis=0, dtype=np.float32)
        proto_vecs.append(m)
        proto_labels.append(clase)
    proto_mat = np.vstack(proto_vecs).astype(np.float32)
    faiss.normalize_L2(proto_mat)
    return proto_mat, proto_labels

def evaluar_closed_set(index, db_labels, query_vecs, query_labels, k_top):
    total = len(query_labels)
    top1_ok, top5_ok, false_accepts = 0, 0, 0
    k_eff = min(k_top, len(db_labels))
    for i in range(total):
        q = query_vecs[i : i + 1]
        sims, idxs = index.search(q, k_eff)
        vecinos = idxs[0]
        labels_pred = [db_labels[j] for j in vecinos if j >= 0]
        if not labels_pred:
            false_accepts += 1
            continue
        if labels_pred[0] == query_labels[i]:
            top1_ok += 1
        else:
            false_accepts += 1
        if query_labels[i] in labels_pred[: min(5, len(labels_pred))]:
            top5_ok += 1
    return {
        "top1": (100.0 * top1_ok / total) if total else 0.0,
        "top5": (100.0 * top5_ok / total) if total else 0.0,
        "false_accepts": int(false_accepts),
        "total_queries": int(total),
    }

def evaluar_open_set(index, db_labels, query_vecs, query_labels, threshold):
    total = len(query_labels)
    accepts_correctos, rechazos, false_accepts = 0, 0, 0
    for i in range(total):
        q = query_vecs[i : i + 1]
        sims, idxs = index.search(q, 1) 
        score = float(sims[0][0])
        idx = int(idxs[0][0])
        
        if idx < 0 or score < threshold:
            rechazos += 1
            continue
            
        if db_labels[idx] == query_labels[i]:
            accepts_correctos += 1
        else:
            false_accepts += 1
            
    return {
        "top1_thresholded": (100.0 * accepts_correctos / total) if total else 0.0,
        "rejection_rate": (100.0 * rechazos / total) if total else 0.0,
        "false_accepts": int(false_accepts),
    }

def imprimir_tablas(closed_all, closed_proto, open_all, open_proto, n_gal_all, n_gal_proto, thr):
    print("======================================================================")
    print(" MVP Cow Tracker - Closed-set (CROSS-POSE STRICT)")
    print("======================================================================")
    print("Métrica              | All-vectors     | Prototype      ")
    print("----------------------------------------------------------------------")
    print(f"Top-1 Accuracy       | {closed_all['top1']:13.2f}% | {closed_proto['top1']:13.2f}%")
    print(f"Top-5 Accuracy       | {closed_all['top5']:13.2f}% | {closed_proto['top5']:13.2f}%")
    print(f"Gallery vectors      | {n_gal_all:14d} | {n_gal_proto:14d}")
    print(f"Query count          | {closed_all['total_queries']:14d} | {closed_proto['total_queries']:14d}")
    print(f"False accepts        | {closed_all['false_accepts']:14d} | {closed_proto['false_accepts']:14d}")
    print()
    print("======================================================================")
    print(f" MVP Cow Tracker - Open-set (Threshold = {thr:.2f})")
    print("======================================================================")
    print("Config               | Top-1 thresholded  | Rejection rate  | False accepts")
    print("----------------------------------------------------------------------")
    print(f"All-vectors + thr    | {open_all['top1_thresholded']:18.2f}% | {open_all['rejection_rate']:13.2f}% | {open_all['false_accepts']:13d}")
    print(f"Prototype + thr      | {open_proto['top1_thresholded']:18.2f}% | {open_proto['rejection_rate']:13.2f}% | {open_proto['false_accepts']:13d}")
    print("======================================================================")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datos/entrenamiento/fotos/")
    parser.add_argument("--model_path", type=str, default="models/mi_modelo_reid.pt")
    parser.add_argument("--threshold", type=float, default=0.85)
    
    # IMPORTANTE: Tomamos un máximo de 50 fotos para Galería y 50 para Query por vaca
    # para que la evaluación en FAISS sea rápida y no colapse tu RAM.
    parser.add_argument("--max_per_cow", type=int, default=50) 
    args = parser.parse_args()

    device = torch.device("cpu")

    agrupado = recolectar_cross_pose(args.data_dir)
    gallery_paths, gallery_labels, query_paths, query_labels = construir_splits_espejo(
        agrupado, max_por_vaca=args.max_per_cow
    )

    model = cargar_modelo(args.model_path, device)
    eval_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    gallery_vecs, gallery_labels_ok = extraer_embeddings(gallery_paths, gallery_labels, model, device, eval_tfm)
    query_vecs, query_labels_ok = extraer_embeddings(query_paths, query_labels, model, device, eval_tfm)

    index_all = faiss.IndexFlatIP(gallery_vecs.shape[1])
    index_all.add(gallery_vecs)

    proto_vecs, proto_labels = calcular_prototipos(gallery_vecs, gallery_labels_ok)
    index_proto = faiss.IndexFlatIP(proto_vecs.shape[1])
    index_proto.add(proto_vecs)

    closed_all = evaluar_closed_set(index_all, gallery_labels_ok, query_vecs, query_labels_ok, 5)
    closed_proto = evaluar_closed_set(index_proto, proto_labels, query_vecs, query_labels_ok, 5)

    open_all = evaluar_open_set(index_all, gallery_labels_ok, query_vecs, query_labels_ok, args.threshold)
    open_proto = evaluar_open_set(index_proto, proto_labels, query_vecs, query_labels_ok, args.threshold)

    imprimir_tablas(closed_all, closed_proto, open_all, open_proto, len(gallery_vecs), len(proto_vecs), args.threshold)

if __name__ == "__main__":
    main()