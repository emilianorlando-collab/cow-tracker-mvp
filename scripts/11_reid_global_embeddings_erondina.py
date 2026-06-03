#!/usr/bin/env python3
"""
Script 11: Re-ID global puro por embeddings para Erondina.

Este script no usa reglas manuales de color ni restricciones por identidad.
Hace Re-ID usando solamente el extractor `mi_modelo_reid.pt` y la galeria
`erondina_gallery_embeddings.npz`, comparando contra todos los embeddings de
galeria en lugar de promedios/prototipos.

Pipeline:
1. Desde el minuto util, detecta y trackea vacas con YOLO + BoT-SORT.
2. Acumula embeddings por tracklet durante todo el video.
3. Fusiona solo tracklets no solapados temporalmente cuando sus embeddings son
   suficientemente parecidos.
4. Asigna Maria/Marta/Margarita una unica vez a identidades globales.
5. Valida que no haya duplicados de una misma identidad conocida en un frame.
6. Renderiza solo si se pide con --render.
"""

import argparse
import importlib.util
import json
import os
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


def load_gallery(gallery_path: str):
    data = np.load(gallery_path, allow_pickle=True)
    vectors = normalize_rows(data["gallery_vectors"].astype(np.float32))
    labels = [label_key(x) for x in data["gallery_labels"].tolist()]
    return vectors, labels


def score_tracklet_to_gallery(tracklet: Tracklet, gallery_vectors: np.ndarray, gallery_labels: List[str]):
    mat = tracklet_embedding_matrix(tracklet)
    if mat is None:
        return {}

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
            "embedding_count": int(len(best_per_observation)),
        }
    return scores


def tracklet_similarity(a: Tracklet, b: Tracklet) -> float:
    ma = tracklet_embedding_matrix(a)
    mb = tracklet_embedding_matrix(b)
    if ma is None or mb is None:
        return -1.0
    sims = ma @ mb.T
    return float(np.percentile(np.max(sims, axis=1), 85))


def can_merge(tracklets: Dict[int, Tracklet], cluster_a: List[int], cluster_b: List[int], max_overlap_frames: int) -> bool:
    for a in cluster_a:
        for b in cluster_b:
            if temporal_overlap(tracklets[a], tracklets[b]) > max_overlap_frames:
                return False
    return True


def cluster_tracklets(tracklets: Dict[int, Tracklet], min_track_frames: int, merge_threshold: float, max_overlap_frames: int):
    valid = [tid for tid, tr in tracklets.items() if tr.frames_seen >= min_track_frames and tr.embeddings]
    clusters = {tid: [tid] for tid in valid}

    def cluster_score(a_ids: List[int], b_ids: List[int]) -> float:
        vals = []
        for a in a_ids:
            for b in b_ids:
                vals.append(tracklet_similarity(tracklets[a], tracklets[b]))
        return float(np.max(vals)) if vals else -1.0

    while True:
        keys = sorted(clusters)
        best_pair = None
        best_score = -1.0
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                if not can_merge(tracklets, clusters[ka], clusters[kb], max_overlap_frames):
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
        best_by_label = {}
        total_frames = sum(tracklets[x].frames_seen for x in local_ids)
        for local_id in local_ids:
            scores = score_tracklet_to_gallery(tracklets[local_id], gallery_vectors, gallery_labels)
            for label, info in scores.items():
                prev = best_by_label.get(label)
                if prev is None or info["score"] > prev["score"]:
                    best_by_label[label] = {**info, "source_local_track_id": local_id}
        if not best_by_label:
            continue
        ranked = sorted(best_by_label.items(), key=lambda x: x[1]["score"], reverse=True)
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
                "source_local_track_id": int(best_info["source_local_track_id"]),
                "cluster_local_track_ids": [int(x) for x in local_ids],
                "cluster_frames": int(total_frames),
                "all_label_scores": best_by_label,
            }
        )
    return candidates


def assign_known_identities(candidates, threshold, margin, min_support):
    candidates = [
        cand for cand in candidates
        if cand["score"] >= threshold and cand["margin"] >= margin and cand["support_075"] >= min_support
    ]

    chosen_by_label = {}
    used_global = set()
    for cand in sorted(candidates, key=lambda x: (x["score"], x["support_075"], x["cluster_frames"]), reverse=True):
        if cand["label"] in chosen_by_label:
            continue
        if cand["global_id"] in used_global:
            continue
        chosen_by_label[cand["label"]] = cand
        used_global.add(cand["global_id"])
    return {cand["global_id"]: cand for cand in chosen_by_label.values()}


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
            cap = cv2.VideoCapture(args.video_in)
            cap.set(cv2.CAP_PROP_POS_FRAMES, rec.frame_number - 1)
            ok, frame = cap.read()
            cap.release()
            if ok:
                crop = base05.recortar_bbox(frame, rec.bbox)
            else:
                crop = np.zeros((180, 220, 3), dtype=np.uint8)
            crop = cv2.resize(crop, (220, 180), interpolation=cv2.INTER_AREA)
            cv2.putText(crop, f"{label} f{rec.frame_number}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
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
            local_ids = set(cand["cluster_local_track_ids"])
            recs = [
                rec for frame in frame_records for rec in frame
                if local_to_global.get(rec.local_track_id) == gid or rec.local_track_id in local_ids
            ]
            if not recs:
                continue
            rec = recs[len(recs) // 2]
            cap = cv2.VideoCapture(args.video_in)
            cap.set(cv2.CAP_PROP_POS_FRAMES, rec.frame_number - 1)
            ok, frame = cap.read()
            cap.release()
            if ok:
                crop = base05.recortar_bbox(frame, rec.bbox)
            else:
                crop = np.zeros((180, 220, 3), dtype=np.uint8)
            crop = cv2.resize(crop, (220, 180), interpolation=cv2.INTER_AREA)
            text = f"{display_label(label)} G{gid} s{score:.2f}"
            cv2.putText(crop, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
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
    cap = cv2.VideoCapture(args.video_in)
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    os.makedirs(os.path.dirname(args.video_out), exist_ok=True)
    writer = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else 30.0, (width, height))
    for idx, records in enumerate(frame_records):
        ok, frame = cap.read()
        if not ok:
            break
        for rec in records:
            gid = local_to_global.get(rec.local_track_id)
            if gid is None:
                continue
            label = label_by_gid.get(gid, f"Vaca {gid:02d}")
            text = f"{label} FIX" if gid in label_by_gid else label
            color = base05.color_for_id(gid)
            x1, y1, x2, y2 = rec.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if gid in label_by_gid else 2)
            cv2.putText(frame, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        writer.write(frame)
        if idx == 0 or (idx + 1) % 100 == 0:
            print(f"Render frame {idx + 1}/{len(frame_records)}")
    cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Re-ID global puro por embeddings")
    parser.add_argument("--video_in", type=str, default="datos/video campo erondina a procesar.MP4")
    parser.add_argument("--video_out", type=str, default="datos/Resultados/resultado_erondina_reid_embeddings_final.mp4")
    parser.add_argument("--report_out", type=str, default="reports/11_reid_global_embeddings_erondina.json")
    parser.add_argument("--contact_sheet_out", type=str, default="reports/11_contact_sheet_embeddings.jpg")
    parser.add_argument("--candidate_sheet_out", type=str, default="reports/11_candidate_sheet_embeddings.jpg")
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
    parser.add_argument("--merge_threshold", type=float, default=0.78)
    parser.add_argument("--max_overlap_frames", type=int, default=5)
    parser.add_argument("--identity_threshold", type=float, default=0.78)
    parser.add_argument("--identity_margin", type=float, default=0.015)
    parser.add_argument("--min_identity_support", type=float, default=0.55)
    parser.add_argument("--expected_total_cows", type=int, default=14)
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
                embeddings = base05.extraer_embeddings_detecciones(frame, reid_bboxes, reid_model, device, tfm)
                for local_id, emb in zip(reid_ids, embeddings):
                    tracklets[local_id].embeddings.append(emb)
        frame_records.append(records)
        if processed == 1 or processed % 100 == 0:
            print(f"Track frame {processed}/{total_frames - args.start_frame}")
    cap.release()

    local_to_global = cluster_tracklets(tracklets, args.min_track_frames, args.merge_threshold, args.max_overlap_frames)
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
    )
    duplicates = duplicate_known_frames(frame_records, local_to_global, known_assignments)
    contact_sheet = make_contact_sheet(args, frame_records, local_to_global, known_assignments, args.contact_sheet_out)
    candidate_sheet = make_candidate_sheet(args, frame_records, local_to_global, all_candidates, args.candidate_sheet_out)

    global_ids = sorted(set(local_to_global.values()))
    found = sorted(data["label"] for data in known_assignments.values())
    missing = sorted(set(known_required) - set(found))
    estimated_total = len(global_ids)
    count_error = estimated_total - args.expected_total_cows
    count_accuracy = max(0.0, 1.0 - abs(count_error) / args.expected_total_cows) if args.expected_total_cows else 0.0
    ready = not missing and not duplicates

    report = {
        "video_in": args.video_in,
        "video_out": args.video_out if args.render else None,
        "processed_frames": processed,
        "estimated_global_cows": estimated_total,
        "expected_total_cows": args.expected_total_cows,
        "count_error": count_error,
        "count_accuracy": count_accuracy,
        "known_found": [display_label(x) for x in found],
        "known_missing": [display_label(x) for x in missing],
        "duplicate_known_label_frames": duplicates,
        "ready_for_render_by_automatic_checks": ready,
        "known_assignments_by_global_id": {str(k): v for k, v in sorted(known_assignments.items())},
        "all_known_candidates": all_candidates,
        "local_to_global": {str(k): int(v) for k, v in sorted(local_to_global.items())},
        "contact_sheet": contact_sheet,
        "candidate_sheet": candidate_sheet,
        "params": vars(args),
    }
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("============================================================")
    print("Reporte Re-ID global puro por embeddings")
    print("============================================================")
    print(f"Frames procesados        : {processed}")
    print(f"Vacas globales estimadas : {estimated_total}")
    print(f"Accuracy conteo estimado    : {100.0 * count_accuracy:.2f}%")
    print(f"Conocidas encontradas    : {', '.join(report['known_found']) if report['known_found'] else 'ninguna'}")
    print(f"Conocidas faltantes      : {', '.join(report['known_missing']) if report['known_missing'] else 'ninguna'}")
    print(f"Duplicados conocidos     : {duplicates}")
    print(f"Checks automáticos OK    : {ready}")
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
