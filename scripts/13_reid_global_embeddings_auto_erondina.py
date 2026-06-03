#!/usr/bin/env python3
"""
Script 13: Re-ID global automatico por embeddings para Erondina.

Este script no usa reglas manuales de color ni mapeos manuales de IDs.
Hace Re-ID usando solamente el extractor `mi_modelo_reid.pt` y la galeria
`erondina_gallery_embeddings.npz`, comparando contra todos los embeddings de
galeria en lugar de promedios/prototipos.

Pipeline:
1. Desde el minuto util, detecta y trackea vacas con YOLO + BoT-SORT.
2. Acumula embeddings por tracklet durante todo el video.
3. Fusiona tracklets no solapados con una similitud robusta y simetrica.
4. Puntua cada identidad global completa, no solo su mejor fragmento.
5. Asigna Maria/Marta/Margarita automaticamente una unica vez.
6. Valida duplicados y genera metricas antes de renderizar.
"""

import argparse
import importlib.util
import itertools
import json
import os
import pickle
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def cargar_modulo(nombre: str, path: str):
    spec = importlib.util.spec_from_file_location(nombre, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base05 = cargar_modulo("base05", os.path.join(SCRIPT_DIR, "05_inferencia_video_erondina.py"))


@dataclass
class FrameBox:
    frame_number: int
    local_track_id: int
    bbox: Tuple[int, int, int, int]
    conf: float


@dataclass
class Tracklet:
    local_track_id: int
    first_frame: int
    last_frame: int
    frames_seen: int = 0
    embeddings: List[np.ndarray] = field(default_factory=list)


def label_key(label: str) -> str:
    text = unicodedata.normalize("NFD", str(label))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.lower()


def display_label(label: str) -> str:
    mapping = {"maria": "Maria", "marta": "Marta", "margarita": "Margarita"}
    return mapping.get(label_key(label), str(label))


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    return mat.astype(np.float32) / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)


def tracklet_embedding_matrix(tracklet: Tracklet):
    if not tracklet.embeddings:
        return None
    return normalize_rows(np.vstack(tracklet.embeddings).astype(np.float32))


def temporal_overlap(a: Tracklet, b: Tracklet) -> int:
    return max(0, min(a.last_frame, b.last_frame) - max(a.first_frame, b.first_frame) + 1)


def temporal_gap(a: Tracklet, b: Tracklet) -> int:
    if temporal_overlap(a, b) > 0:
        return 0
    return max(a.first_frame, b.first_frame) - min(a.last_frame, b.last_frame) - 1


def load_gallery(gallery_path: str):
    data = np.load(gallery_path, allow_pickle=True)
    vectors = normalize_rows(data["gallery_vectors"].astype(np.float32))
    labels = [label_key(x) for x in data["gallery_labels"].tolist()]
    return vectors, labels


def score_matrix_to_gallery(mat: np.ndarray, gallery_vectors: np.ndarray, gallery_labels: List[str]):
    if mat is None or len(mat) == 0:
        return {}, {}
    mat = normalize_rows(mat)
    all_sims = mat @ gallery_vectors.T
    nearest_idxs = np.argmax(all_sims, axis=1)
    nearest_labels = [gallery_labels[int(i)] for i in nearest_idxs]
    vote_fractions = {
        label: float(sum(1 for item in nearest_labels if item == label) / len(nearest_labels))
        for label in sorted(set(gallery_labels))
    }
    scores = {}
    for label in sorted(set(gallery_labels)):
        idxs = [i for i, lab in enumerate(gallery_labels) if lab == label]
        if not idxs:
            continue
        sims = mat @ gallery_vectors[idxs].T
        best_per_observation = np.max(sims, axis=1)
        scores[label] = {
            "score": float(np.percentile(best_per_observation, 85)),
            "median": float(np.median(best_per_observation)),
            "max": float(np.max(best_per_observation)),
            "support_075": float(np.mean(best_per_observation >= 0.75)),
            "support_080": float(np.mean(best_per_observation >= 0.80)),
            "nearest_vote_fraction": float(vote_fractions.get(label, 0.0)),
            "embedding_count": int(len(best_per_observation)),
        }
    return scores, vote_fractions


def score_tracklet_to_gallery(tracklet: Tracklet, gallery_vectors: np.ndarray, gallery_labels: List[str]):
    scores, _ = score_matrix_to_gallery(tracklet_embedding_matrix(tracklet), gallery_vectors, gallery_labels)
    return scores


def cluster_embedding_matrix(tracklets: Dict[int, Tracklet], local_ids: List[int]):
    mats = []
    for local_id in local_ids:
        mat = tracklet_embedding_matrix(tracklets[local_id])
        if mat is not None:
            mats.append(mat)
    if not mats:
        return None
    return normalize_rows(np.vstack(mats).astype(np.float32))


def tracklet_similarity(a: Tracklet, b: Tracklet) -> float:
    ma = tracklet_embedding_matrix(a)
    mb = tracklet_embedding_matrix(b)
    if ma is None or mb is None:
        return -1.0
    sims = ma @ mb.T
    a_to_b = np.percentile(np.max(sims, axis=1), 75)
    b_to_a = np.percentile(np.max(sims, axis=0), 75)
    return float(min(a_to_b, b_to_a))


def can_merge(
    tracklets: Dict[int, Tracklet],
    cluster_a: List[int],
    cluster_b: List[int],
    max_overlap_frames: int,
    max_merge_gap_frames: int,
) -> bool:
    for a in cluster_a:
        for b in cluster_b:
            if temporal_overlap(tracklets[a], tracklets[b]) > max_overlap_frames:
                return False
            if max_merge_gap_frames > 0 and temporal_gap(tracklets[a], tracklets[b]) > max_merge_gap_frames:
                return False
    return True


def cluster_tracklets(
    tracklets: Dict[int, Tracklet],
    min_track_frames: int,
    merge_threshold: float,
    max_overlap_frames: int,
    max_merge_gap_frames: int,
):
    valid = [tid for tid, tr in tracklets.items() if tr.frames_seen >= min_track_frames and tr.embeddings]
    clusters = {tid: [tid] for tid in valid}

    def cluster_score(a_ids: List[int], b_ids: List[int]) -> float:
        vals = []
        for a in a_ids:
            for b in b_ids:
                vals.append(tracklet_similarity(tracklets[a], tracklets[b]))
        if not vals:
            return -1.0
        vals = np.array(vals, dtype=np.float32)
        return float(np.percentile(vals, 65))

    while True:
        keys = sorted(clusters)
        best_pair = None
        best_score = -1.0
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                if not can_merge(tracklets, clusters[ka], clusters[kb], max_overlap_frames, max_merge_gap_frames):
                    continue
                score = cluster_score(clusters[ka], clusters[kb])
                if score > best_score:
                    best_score = score
                    best_pair = (ka, kb)
        if best_pair is None or best_score < merge_threshold:
            break
        keep, drop = best_pair
        clusters[keep].extend(clusters[drop])
        del clusters[drop]

    local_to_global = {}
    for global_id, key in enumerate(sorted(clusters), start=1):
        for local_id in clusters[key]:
            local_to_global[local_id] = global_id
    return local_to_global


def collect_known_candidates(tracklets, local_to_global, gallery_vectors, gallery_labels):
    grouped = defaultdict(list)
    for local_id, global_id in local_to_global.items():
        grouped[global_id].append(local_id)

    candidates = []
    for global_id, local_ids in grouped.items():
        total_frames = sum(tracklets[x].frames_seen for x in local_ids)
        cluster_mat = cluster_embedding_matrix(tracklets, local_ids)
        cluster_scores, vote_fractions = score_matrix_to_gallery(cluster_mat, gallery_vectors, gallery_labels)
        if not cluster_scores:
            continue

        best_sources = {}
        for label in sorted(set(gallery_labels)):
            best_source = None
            for local_id in local_ids:
                scores = score_tracklet_to_gallery(tracklets[local_id], gallery_vectors, gallery_labels)
                info = scores.get(label)
                if not info:
                    continue
                if best_source is None or info["score"] > best_source["score"]:
                    best_source = {**info, "source_local_track_id": local_id}
            if best_source is not None:
                best_sources[label] = best_source

        enriched_scores = {}
        for label, info in cluster_scores.items():
            source = best_sources.get(label, {})
            enriched_scores[label] = {
                **info,
                "source_local_track_id": int(source.get("source_local_track_id", local_ids[0])),
                "source_score": float(source.get("score", info["score"])),
                "cluster_score_mode": "all_embeddings",
            }

        ranked = sorted(enriched_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        best_label, best_info = ranked[0]
        second_score = ranked[1][1]["score"] if len(ranked) > 1 else -1.0
        best_margin = best_info["score"] - second_score
        candidates.append(
            {
                "global_id": global_id,
                "label": best_label,
                "score": float(best_info["score"]),
                "margin": float(best_margin),
                "median": float(best_info["median"]),
                "max": float(best_info["max"]),
                "support_075": float(best_info["support_075"]),
                "support_080": float(best_info["support_080"]),
                "nearest_vote_fraction": float(best_info["nearest_vote_fraction"]),
                "source_local_track_id": int(best_info["source_local_track_id"]),
                "cluster_local_track_ids": [int(x) for x in local_ids],
                "cluster_frames": int(total_frames),
                "all_label_scores": enriched_scores,
                "nearest_vote_fractions": vote_fractions,
            }
        )
    return candidates


def label_candidate(cand, label, threshold, margin, min_support, min_vote_fraction):
    info = cand.get("all_label_scores", {}).get(label)
    if not info:
        return None
    other_scores = [
        other["score"]
        for other_label, other in cand.get("all_label_scores", {}).items()
        if other_label != label
    ]
    second = max(other_scores) if other_scores else -1.0
    label_margin = float(info["score"] - second)
    if info["score"] < threshold:
        return None
    if info["support_075"] < min_support:
        return None
    if info["nearest_vote_fraction"] < min_vote_fraction:
        return None
    if label_margin < margin:
        return None

    quality = (
        float(info["score"])
        + 0.08 * float(info["support_075"])
        + 0.08 * float(info["nearest_vote_fraction"])
        + 0.04 * max(0.0, label_margin)
        + 0.00001 * min(5000, int(cand["cluster_frames"]))
    )
    return {
        "global_id": int(cand["global_id"]),
        "label": label,
        "score": float(info["score"]),
        "margin": label_margin,
        "median": float(info["median"]),
        "max": float(info["max"]),
        "support_075": float(info["support_075"]),
        "support_080": float(info["support_080"]),
        "nearest_vote_fraction": float(info["nearest_vote_fraction"]),
        "source_local_track_id": int(info["source_local_track_id"]),
        "source_score": float(info.get("source_score", info["score"])),
        "cluster_local_track_ids": [int(x) for x in cand["cluster_local_track_ids"]],
        "cluster_frames": int(cand["cluster_frames"]),
        "all_label_scores": cand["all_label_scores"],
        "automatic_assignment_quality": float(quality),
        "assignment_mode": "automatic_global_all_embeddings",
    }


def assign_known_identities(candidates, threshold, margin, min_support, min_vote_fraction):
    labels = sorted({label for cand in candidates for label in cand.get("all_label_scores", {}).keys()})
    per_label = {}
    for label in labels:
        rows = []
        for cand in candidates:
            item = label_candidate(cand, label, threshold, margin, min_support, min_vote_fraction)
            if item is not None:
                rows.append(item)
        per_label[label] = sorted(
            rows,
            key=lambda x: (x["automatic_assignment_quality"], x["score"], x["cluster_frames"]),
            reverse=True,
        )[:12]

    labels_with_options = [label for label in labels if per_label[label]]
    best_combo = []
    best_quality = -1.0
    for combo in itertools.product(*(per_label[label] for label in labels_with_options)):
        gids = [item["global_id"] for item in combo]
        if len(set(gids)) != len(gids):
            continue
        quality = float(sum(item["automatic_assignment_quality"] for item in combo))
        if quality > best_quality:
            best_quality = quality
            best_combo = list(combo)

    return {item["global_id"]: item for item in best_combo}


def duplicate_known_frames(frame_records, local_to_global, known_assignments):
    label_by_gid = {gid: data["label"] for gid, data in known_assignments.items()}
    duplicates = defaultdict(int)
    for records in frame_records:
        labels = defaultdict(int)
        for rec in records:
            gid = local_to_global.get(rec.local_track_id)
            if gid in label_by_gid:
                labels[label_by_gid[gid]] += 1
        for label, count in labels.items():
            if count > 1:
                duplicates[label] += 1
    return dict(duplicates)


def read_crop(video_path: str, rec: FrameBox, size=(260, 200)):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, rec.frame_number - 1)
    ok, frame = cap.read()
    cap.release()
    if ok:
        crop = base05.recortar_bbox(frame, rec.bbox)
    else:
        crop = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def focus_bgr_crop(crop_bgr: np.ndarray, margin_x: float, margin_y: float) -> np.ndarray:
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    x1 = int(round(w * margin_x))
    y1 = int(round(h * margin_y))
    x2 = int(round(w * (1.0 - margin_x)))
    y2 = int(round(h * (1.0 - margin_y)))
    if x2 <= x1 or y2 <= y1:
        return crop_bgr
    return crop_bgr[y1:y2, x1:x2]


def extraer_embeddings_detecciones_enfocadas(frame_bgr, bboxes, reid_model, device, transform, margin_x, margin_y):
    embeddings = []
    with base05.torch.no_grad():
        for bbox in bboxes:
            crop_bgr = base05.recortar_bbox(frame_bgr, bbox)
            crop_bgr = focus_bgr_crop(crop_bgr, margin_x, margin_y)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            x = transform(crop_rgb).unsqueeze(0).to(device)
            emb = reid_model(x).cpu().numpy()[0].astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings.append(emb)
    return embeddings


def draw_text_bg(img, text, org, font_scale, color, thickness=2):
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x2 = min(img.shape[1] - 1, x + tw + 8)
    y1 = max(0, y - th - baseline - 8)
    y2 = min(img.shape[0] - 1, y + baseline + 5)
    cv2.rectangle(img, (x, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 4, y), font, font_scale, color, thickness, cv2.LINE_AA)


def make_contact_sheet(args, frame_records, local_to_global, known_assignments, out_path):
    gid_to_label = {gid: display_label(data["label"]) for gid, data in known_assignments.items()}
    rows = []
    for gid, label in sorted(gid_to_label.items(), key=lambda x: x[1]):
        recs = [rec for frame in frame_records for rec in frame if local_to_global.get(rec.local_track_id) == gid]
        if not recs:
            continue
        picks = [recs[0], recs[len(recs) // 2], recs[-1]]
        cells = []
        for rec in picks:
            crop = read_crop(args.video_in, rec)
            draw_text_bg(crop, f"{label} G{gid} f{rec.frame_number}", (6, 28), 0.62, (0, 255, 255), 2)
            cells.append(crop)
        rows.append(np.hstack(cells))
    if not rows:
        return None
    sheet = np.vstack(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, sheet)
    return out_path


def make_candidate_sheet(args, frame_records, local_to_global, candidates, out_path, top_k=4):
    rows = []
    labels = sorted({label for cand in candidates for label in cand.get("all_label_scores", {}).keys()})
    for label in labels:
        ranked = []
        for cand in candidates:
            info = cand.get("all_label_scores", {}).get(label)
            if not info:
                continue
            ranked.append((info["score"], cand["global_id"], cand, info))
        ranked = sorted(ranked, reverse=True)[:top_k]

        cells = []
        for score, gid, cand, info in ranked:
            source_local_id = int(info.get("source_local_track_id", cand["cluster_local_track_ids"][0]))
            recs = [
                rec for frame in frame_records for rec in frame
                if rec.local_track_id == source_local_id
            ]
            if not recs:
                continue
            rec = recs[len(recs) // 2]
            crop = read_crop(args.video_in, rec)
            text = f"{display_label(label)} G{gid} L{source_local_id} s{score:.2f}"
            draw_text_bg(crop, text, (6, 28), 0.54, (0, 255, 255), 2)
            cells.append(crop)
        if cells:
            while len(cells) < top_k:
                cells.append(np.zeros_like(cells[0]))
            rows.append(np.hstack(cells))

    if not rows:
        return None
    sheet = np.vstack(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, sheet)
    return out_path


def render_video(args, frame_records, local_to_global, known_assignments, width, height, fps):
    label_by_gid = {gid: display_label(data["label"]) for gid, data in known_assignments.items()}
    score_by_gid = {gid: float(data["score"]) for gid, data in known_assignments.items()}
    cap = cv2.VideoCapture(args.video_in)
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    os.makedirs(os.path.dirname(args.video_out), exist_ok=True)
    writer = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else 30.0, (width, height))
    for idx, records in enumerate(frame_records):
        ok, frame = cap.read()
        if not ok:
            break
        visible_global_ids = set()
        for rec in records:
            gid = local_to_global.get(rec.local_track_id)
            if gid is None:
                continue
            visible_global_ids.add(gid)
            color = base05.color_for_id(gid)
            x1, y1, x2, y2 = rec.bbox
            if gid in label_by_gid:
                text = f"{label_by_gid[gid]} G{gid:02d} {score_by_gid[gid]:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                draw_text_bg(frame, text, (x1, max(42, y1 - 12)), 1.05, color, 3)
            else:
                text = f"Vaca G{gid:02d}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                draw_text_bg(frame, text, (x1, max(34, y1 - 10)), 0.82, color, 2)
        overlay = f"Frame {idx + 1}/{len(frame_records)} | visibles {len(visible_global_ids)} | globales {len(set(local_to_global.values()))}"
        draw_text_bg(frame, overlay, (28, 52), 0.9, (255, 255, 255), 2)
        writer.write(frame)
        if idx == 0 or (idx + 1) % 100 == 0:
            print(f"Render frame {idx + 1}/{len(frame_records)}")
    cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Re-ID global automatico por embeddings")
    parser.add_argument("--video_in", type=str, default="datos/video campo erondina a procesar.MP4")
    parser.add_argument("--video_out", type=str, default="datos/Resultados/resultado_erondina_reid_auto_final.mp4")
    parser.add_argument("--report_out", type=str, default="reports/13_reid_auto_final.json")
    parser.add_argument("--contact_sheet_out", type=str, default="reports/13_contact_sheet_auto_final.jpg")
    parser.add_argument("--candidate_sheet_out", type=str, default="reports/13_candidate_sheet_auto_final.jpg")
    parser.add_argument("--yolo_model", type=str, default="scripts/yolov8m.pt")
    parser.add_argument("--tracker", type=str, default="botsort.yaml")
    parser.add_argument("--reid_model", type=str, default="models/mi_modelo_reid.pt")
    parser.add_argument("--identity_gallery", type=str, default="models/erondina_gallery_embeddings.npz")
    parser.add_argument("--start_frame", type=int, default=1800)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--det_conf", type=float, default=0.18)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--reid_every", type=int, default=8)
    parser.add_argument("--min_track_frames", type=int, default=200)
    parser.add_argument("--merge_threshold", type=float, default=0.82)
    parser.add_argument("--max_overlap_frames", type=int, default=5)
    parser.add_argument("--max_merge_gap_frames", type=int, default=1800)
    parser.add_argument("--identity_threshold", type=float, default=0.78)
    parser.add_argument("--identity_margin", type=float, default=0.02)
    parser.add_argument("--min_identity_support", type=float, default=0.30)
    parser.add_argument("--min_identity_vote_fraction", type=float, default=0.40)
    parser.add_argument("--expected_total_cows", type=int, default=14)
    parser.add_argument("--count_tolerance", type=int, default=2)
    parser.add_argument("--focus_margin_x", type=float, default=0.0)
    parser.add_argument("--focus_margin_y", type=float, default=0.0)
    parser.add_argument("--evidence_cache_in", type=str, default="")
    parser.add_argument("--evidence_cache_out", type=str, default="")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    device = base05.torch.device("cpu")
    detector = YOLO(args.yolo_model)
    cow_class_id = base05.obtener_cow_class_id(detector)
    reid_model = base05.cargar_reid_model(args.reid_model, device)
    gallery_vectors, gallery_labels = load_gallery(args.identity_gallery)
    known_required = sorted(set(gallery_labels))
    tfm = base05.transforms.Compose(
        [
            base05.transforms.ToPILImage(),
            base05.transforms.Resize((224, 224)),
            base05.transforms.ToTensor(),
            base05.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if args.evidence_cache_in:
        print(f"Cargando evidencia cacheada: {args.evidence_cache_in}")
        with open(args.evidence_cache_in, "rb") as f:
            payload = pickle.load(f)
        tracklets = payload["tracklets"]
        frame_records = payload["frame_records"]
        processed = int(payload["processed"])
        total_frames = int(payload["total_frames"])
        fps = float(payload["fps"])
        width = int(payload["width"])
        height = int(payload["height"])
    else:
        cap = cv2.VideoCapture(args.video_in)
        if not cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el video: {args.video_in}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if args.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

        tracklets: Dict[int, Tracklet] = {}
        frame_records: List[List[FrameBox]] = []
        processed = 0
        frame_number = args.start_frame
        print("Pass 1/2: tracking + evidencia ReID por embeddings")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            processed += 1
            frame_number += 1
            if args.max_frames > 0 and processed > args.max_frames:
                break
            result = detector.track(
                source=frame,
                persist=True,
                tracker=args.tracker,
                conf=args.det_conf,
                imgsz=args.imgsz,
                verbose=False,
                device="cpu",
                classes=[cow_class_id],
            )[0]
            records = []
            if result.boxes is not None and result.boxes.id is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                reid_bboxes = []
                reid_ids = []
                for box, local_id, conf in zip(boxes, ids, confs):
                    local_id = int(local_id)
                    bbox = tuple(map(int, box.tolist()))
                    records.append(FrameBox(frame_number=frame_number, local_track_id=local_id, bbox=bbox, conf=float(conf)))
                    if local_id not in tracklets:
                        tracklets[local_id] = Tracklet(local_track_id=local_id, first_frame=frame_number, last_frame=frame_number)
                    tr = tracklets[local_id]
                    tr.last_frame = frame_number
                    tr.frames_seen += 1
                    if processed % args.reid_every == 0:
                        reid_bboxes.append(bbox)
                        reid_ids.append(local_id)
                if reid_bboxes:
                    if args.focus_margin_x > 0 or args.focus_margin_y > 0:
                        embeddings = extraer_embeddings_detecciones_enfocadas(
                            frame,
                            reid_bboxes,
                            reid_model,
                            device,
                            tfm,
                            args.focus_margin_x,
                            args.focus_margin_y,
                        )
                    else:
                        embeddings = base05.extraer_embeddings_detecciones(frame, reid_bboxes, reid_model, device, tfm)
                    for local_id, emb in zip(reid_ids, embeddings):
                        tracklets[local_id].embeddings.append(emb)
            frame_records.append(records)
            if processed == 1 or processed % 100 == 0:
                print(f"Track frame {processed}/{total_frames - args.start_frame}")
        cap.release()

        if args.evidence_cache_out:
            os.makedirs(os.path.dirname(args.evidence_cache_out), exist_ok=True)
            with open(args.evidence_cache_out, "wb") as f:
                pickle.dump(
                    {
                        "tracklets": tracklets,
                        "frame_records": frame_records,
                        "processed": processed,
                        "total_frames": total_frames,
                        "fps": fps,
                        "width": width,
                        "height": height,
                        "start_frame": args.start_frame,
                        "video_in": args.video_in,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(f"Evidencia guardada en: {args.evidence_cache_out}")

    local_to_global = cluster_tracklets(
        tracklets,
        args.min_track_frames,
        args.merge_threshold,
        args.max_overlap_frames,
        args.max_merge_gap_frames,
    )
    all_candidates = collect_known_candidates(
        tracklets,
        local_to_global,
        gallery_vectors,
        gallery_labels,
    )
    known_assignments = assign_known_identities(
        all_candidates,
        threshold=args.identity_threshold,
        margin=args.identity_margin,
        min_support=args.min_identity_support,
        min_vote_fraction=args.min_identity_vote_fraction,
    )
    duplicates = duplicate_known_frames(frame_records, local_to_global, known_assignments)
    contact_sheet = make_contact_sheet(args, frame_records, local_to_global, known_assignments, args.contact_sheet_out)
    candidate_sheet = make_candidate_sheet(args, frame_records, local_to_global, all_candidates, args.candidate_sheet_out)

    global_ids = sorted(set(local_to_global.values()))
    found = sorted(data["label"] for data in known_assignments.values())
    missing = sorted(set(known_required) - set(found))
    estimated_total = len(global_ids)
    visible_counts = [
        len({local_to_global[rec.local_track_id] for rec in records if rec.local_track_id in local_to_global})
        for records in frame_records
    ]
    visible_mean = float(np.mean(visible_counts)) if visible_counts else 0.0
    visible_median = float(np.median(visible_counts)) if visible_counts else 0.0
    visible_p95 = float(np.percentile(visible_counts, 95)) if visible_counts else 0.0
    visible_max = int(max(visible_counts)) if visible_counts else 0
    estimated_visible_total = int(round(visible_p95)) if visible_counts else estimated_total
    known_frame_hits = defaultdict(int)
    known_gid_to_label = {gid: data["label"] for gid, data in known_assignments.items()}
    for records in frame_records:
        seen_labels = set()
        for rec in records:
            gid = local_to_global.get(rec.local_track_id)
            if gid in known_gid_to_label:
                seen_labels.add(known_gid_to_label[gid])
        for label in seen_labels:
            known_frame_hits[label] += 1
    count_error = estimated_visible_total - args.expected_total_cows
    absolute_count_error = abs(count_error)
    count_accuracy = max(0.0, 1.0 - abs(count_error) / args.expected_total_cows) if args.expected_total_cows else 0.0
    count_within_tolerance = absolute_count_error <= args.count_tolerance
    ready = not missing and not duplicates and count_within_tolerance

    report = {
        "video_in": args.video_in,
        "video_out": args.video_out if args.render else None,
        "processed_frames": processed,
        "estimated_total_cows": estimated_visible_total,
        "estimated_total_cows_method": "visible_cows_p95_per_frame",
        "global_track_count_after_clustering": estimated_total,
        "fragmentation_over_expected": max(0, estimated_total - args.expected_total_cows),
        "expected_total_cows": args.expected_total_cows,
        "count_error": count_error,
        "absolute_count_error": absolute_count_error,
        "count_accuracy": count_accuracy,
        "count_within_tolerance": count_within_tolerance,
        "unknown_cows_estimated": max(0, estimated_visible_total - len(found)),
        "visible_cows_per_frame": {
            "mean": visible_mean,
            "median": visible_median,
            "p95": visible_p95,
            "max": visible_max,
        },
        "known_found": [display_label(x) for x in found],
        "known_missing": [display_label(x) for x in missing],
        "known_frame_hits": {display_label(k): int(v) for k, v in sorted(known_frame_hits.items())},
        "duplicate_known_label_frames": duplicates,
        "known_id_switches_by_design": 0,
        "ready_for_render_by_automatic_checks": ready,
        "known_assignments_by_global_id": {str(k): v for k, v in sorted(known_assignments.items())},
        "all_known_candidates": all_candidates,
        "local_to_global": {str(k): int(v) for k, v in sorted(local_to_global.items())},
        "contact_sheet": contact_sheet,
        "candidate_sheet": candidate_sheet,
        "metric_note": (
            "Precision/recall reales por bounding box requieren anotaciones ground truth por frame. "
            "Este JSON reporta metricas operativas automaticas: conteo vs 14, estabilidad de IDs conocidos, "
            "duplicados de identidad por frame, margen/pureza de ReID, vacas visibles por frame y fragmentacion de tracks."
        ),
        "params": vars(args),
    }
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("============================================================")
    print("Reporte Re-ID global automatico por embeddings")
    print("============================================================")
    print(f"Frames procesados        : {processed}")
    print(f"Vacas estimadas por frame: {estimated_visible_total}")
    print(f"Tracks globales cluster  : {estimated_total}")
    print(f"Accuracy conteo vs 14    : {100.0 * count_accuracy:.2f}%")
    print(f"Error absoluto conteo    : {absolute_count_error}")
    print(f"Vacas visibles mediana   : {report['visible_cows_per_frame']['median']:.2f}")
    print(f"Conocidas encontradas    : {', '.join(report['known_found']) if report['known_found'] else 'ninguna'}")
    print(f"Conocidas faltantes      : {', '.join(report['known_missing']) if report['known_missing'] else 'ninguna'}")
    print(f"Duplicados conocidos     : {duplicates}")
    print(f"Checks automaticos OK    : {ready}")
    print(f"Reporte guardado en      : {args.report_out}")
    if contact_sheet:
        print(f"Contacto visual guardado : {contact_sheet}")
    if candidate_sheet:
        print(f"Candidatas visual guardado: {candidate_sheet}")

    if args.render:
        print("Pass 2/2: render con IDs globales ReID puros")
        render_video(args, frame_records, local_to_global, known_assignments, width, height, fps)
        print(f"Video guardado en        : {args.video_out}")


if __name__ == "__main__":
    main()
