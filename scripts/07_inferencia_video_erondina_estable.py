#!/usr/bin/env python3
"""
Script 07: Inferencia Erondina estable.

Versión experimental para cerrar el MVP sin renderizar a ciegas:
- confirma tracks antes de contarlos/dibujarlos;
- bloquea identidades conocidas una vez confirmadas;
- evita que María/Marta/Margarita cambien de vaca durante el track;
- permite correr previews o métricas sin render completo.
"""

import argparse
import importlib.util
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def cargar_modulo(nombre: str, path: str):
    spec = importlib.util.spec_from_file_location(nombre, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base05 = cargar_modulo("base05", os.path.join(SCRIPT_DIR, "05_inferencia_video_erondina.py"))


@dataclass
class StableTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    embedding: np.ndarray
    first_seen: int
    last_seen: int
    hits: int = 1
    missed: int = 0
    identity_label: str = "desconocida"
    identity_score: float = 0.0
    locked_identity: bool = False
    evidence: Counter = field(default_factory=Counter)


def filtrar_geometria(
    bboxes: List[Tuple[int, int, int, int]],
    confs: List[float],
    frame_w: int,
    frame_h: int,
    min_area_ratio: float,
    max_area_ratio: float,
    min_aspect: float,
    max_aspect: float,
) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    out_boxes = []
    out_confs = []
    frame_area = frame_w * frame_h
    for box, conf in zip(bboxes, confs):
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area_ratio = (bw * bh) / frame_area
        aspect = bw / bh
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        if aspect < min_aspect or aspect > max_aspect:
            continue
        out_boxes.append(box)
        out_confs.append(conf)
    return out_boxes, out_confs


def construir_costos_estables(
    detections: List[Dict],
    tracks: List[StableTrack],
    frame_shape: Tuple[int, int],
    min_cosine: float,
    max_spatial_ratio: float,
) -> np.ndarray:
    if not detections or not tracks:
        return np.zeros((len(detections), len(tracks)), dtype=np.float32)

    h, w = frame_shape
    diag = float(np.hypot(w, h)) + 1e-8
    max_spatial = max_spatial_ratio * diag
    C = np.full((len(detections), len(tracks)), 1e6, dtype=np.float32)

    for i, det in enumerate(detections):
        for j, tr in enumerate(tracks):
            d = float(np.hypot(det["centroid"][0] - tr.centroid[0], det["centroid"][1] - tr.centroid[1]))
            cos = base05.cosine_similarity_np(det["embedding"], tr.embedding)

            # Si está cerca espacialmente, priorizamos continuidad del track.
            if d <= 0.035 * diag:
                C[i, j] = 0.02 + 0.20 * (d / diag) + 0.05 * (1.0 - cos)
                continue

            if d > max_spatial:
                continue
            if cos < min_cosine:
                continue

            C[i, j] = 0.65 * (d / diag) + 0.35 * (1.0 - cos)

    return C


def reconocer_con_bloqueo(
    embedding: np.ndarray,
    gallery,
    threshold: float,
    locked_labels: set,
) -> Tuple[str, float]:
    label, score = base05.reconocer_identidad(embedding, gallery, threshold)
    if label in locked_labels:
        return "desconocida", score
    return label, score


def actualizar_identidad(
    track: StableTrack,
    gallery,
    threshold: float,
    lock_hits: int,
    locked_labels: set,
) -> None:
    if track.locked_identity:
        return

    label, score = reconocer_con_bloqueo(track.embedding, gallery, threshold, locked_labels)
    track.identity_score = score
    if label == "desconocida":
        track.identity_label = "desconocida"
        return

    track.evidence[label] += 1
    best_label, best_hits = track.evidence.most_common(1)[0]
    if best_hits >= lock_hits:
        track.identity_label = best_label
        track.locked_identity = True
        track.identity_score = score
    else:
        track.identity_label = "desconocida"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia estable Erondina con identidad bloqueada")
    parser.add_argument("--video_in", type=str, default="datos/video campo erondina a procesar.MP4")
    parser.add_argument("--video_out", type=str, default="datos/Resultados/resultado_erondina_reid_estable_preview.mp4")
    parser.add_argument("--report_out", type=str, default="reports/07_inferencia_erondina_estable.json")
    parser.add_argument("--yolo_model", type=str, default="scripts/yolov8m.pt")
    parser.add_argument("--reid_model", type=str, default="models/mi_modelo_reid.pt")
    parser.add_argument("--identity_gallery", type=str, default="models/erondina_gallery_embeddings.npz")
    parser.add_argument("--identity_mode", type=str, default="prototype", choices=["prototype", "all"])
    parser.add_argument("--identity_threshold", type=float, default=0.78)
    parser.add_argument("--identity_lock_hits", type=int, default=8)
    parser.add_argument("--det_conf", type=float, default=0.25)
    parser.add_argument("--sim_threshold", type=float, default=0.35)
    parser.add_argument("--max_spatial_ratio", type=float, default=0.35)
    parser.add_argument("--iou_threshold", type=float, default=0.70)
    parser.add_argument("--ema_alpha", type=float, default=0.15)
    parser.add_argument("--min_track_hits", type=int, default=8)
    parser.add_argument("--max_missed", type=int, default=180)
    parser.add_argument("--expected_total_cows", type=int, default=14)
    parser.add_argument("--min_area_ratio", type=float, default=0.00008)
    parser.add_argument("--max_area_ratio", type=float, default=0.02500)
    parser.add_argument("--min_aspect", type=float, default=0.30)
    parser.add_argument("--max_aspect", type=float, default=4.00)
    parser.add_argument("--start_frame", type=int, default=500)
    parser.add_argument("--max_frames", type=int, default=600)
    parser.add_argument("--no_render", action="store_true")
    args = parser.parse_args()

    device = base05.torch.device("cpu")
    detector = YOLO(args.yolo_model)
    cow_class_id = base05.obtener_cow_class_id(detector)
    reid_model = base05.cargar_reid_model(args.reid_model, device)
    gallery = base05.cargar_galeria_identidades(args.identity_gallery, args.identity_mode)
    reid_transform = base05.transforms.Compose(
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

    writer = None
    if not args.no_render:
        os.makedirs(os.path.dirname(args.video_out), exist_ok=True)
        writer = cv2.VideoWriter(
            args.video_out,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else 30.0,
            (width, height),
        )

    tracks: Dict[int, StableTrack] = {}
    next_id = 1
    processed = 0
    frame_idx = args.start_frame
    id_switches = 0
    prev_assignments: List[Tuple[Tuple[int, int, int, int], int]] = []
    frame_counts = []
    identity_hits = defaultdict(int)
    unknown_hits = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        processed += 1
        if args.max_frames > 0 and processed > args.max_frames:
            break
        if processed == 1 or processed % 50 == 0:
            print(f"Frame {frame_idx}/{total_frames}")

        result = detector.predict(source=frame, conf=args.det_conf, imgsz=1280, verbose=False, device="cpu")[0]
        bboxes = []
        confs = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            for box, cls_id, score in zip(xyxy, cls, scores):
                if cls_id != cow_class_id or score < args.det_conf:
                    continue
                bboxes.append(tuple(map(int, box.tolist())))
                confs.append(float(score))

        bboxes, confs = base05.nms_manual(bboxes, confs, iou_thr=args.iou_threshold)
        bboxes, confs = base05.filtrar_anti_mega_cajas(bboxes, confs, frame_w=width, frame_h=height)
        bboxes, confs = filtrar_geometria(
            bboxes,
            confs,
            frame_w=width,
            frame_h=height,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
            min_aspect=args.min_aspect,
            max_aspect=args.max_aspect,
        )

        embeddings = base05.extraer_embeddings_detecciones(frame, bboxes, reid_model, device, reid_transform)
        detections = [
            {"bbox": bbox, "centroid": base05.bbox_centroid(bbox), "embedding": emb, "conf": conf}
            for bbox, emb, conf in zip(bboxes, embeddings, confs)
        ]

        active_ids = [tid for tid, tr in tracks.items() if tr.missed <= args.max_missed]
        active_tracks = [tracks[tid] for tid in active_ids]
        assigned_dets = set()
        assigned_tracks = set()

        if detections and active_tracks:
            C = construir_costos_estables(
                detections,
                active_tracks,
                frame_shape=(height, width),
                min_cosine=args.sim_threshold,
                max_spatial_ratio=args.max_spatial_ratio,
            )
            rows, cols = linear_sum_assignment(C)
            for r, c in zip(rows, cols):
                if C[r, c] >= 1e5:
                    continue
                det = detections[r]
                tr = active_tracks[c]
                new_emb = (1.0 - args.ema_alpha) * tr.embedding + args.ema_alpha * det["embedding"]
                new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-8)
                tr.embedding = new_emb
                tr.bbox = det["bbox"]
                tr.centroid = det["centroid"]
                tr.last_seen = frame_idx
                tr.missed = 0
                tr.hits += 1
                assigned_dets.add(r)
                assigned_tracks.add(tr.track_id)

        locked_labels = {tr.identity_label for tr in tracks.values() if tr.locked_identity}
        for tr in tracks.values():
            if tr.track_id in assigned_tracks:
                actualizar_identidad(tr, gallery, args.identity_threshold, args.identity_lock_hits, locked_labels)
                if tr.locked_identity:
                    locked_labels.add(tr.identity_label)

        for i, det in enumerate(detections):
            if i in assigned_dets:
                continue
            tr_id = next_id
            next_id += 1
            tracks[tr_id] = StableTrack(
                track_id=tr_id,
                bbox=det["bbox"],
                centroid=det["centroid"],
                embedding=det["embedding"],
                first_seen=frame_idx,
                last_seen=frame_idx,
            )

        for tr_id, tr in list(tracks.items()):
            if tr.last_seen != frame_idx:
                tr.missed += 1
            if tr.missed > args.max_missed:
                del tracks[tr_id]

        drawable = [
            tr for tr in tracks.values()
            if tr.hits >= args.min_track_hits and tr.missed <= args.max_missed
        ]
        frame_counts.append(len(drawable))

        current_assignments = []
        for tr in drawable:
            x1, y1, x2, y2 = tr.bbox
            color = base05.color_for_id(tr.track_id)
            current_assignments.append((tr.bbox, tr.track_id))
            if tr.identity_label == "desconocida":
                unknown_hits += 1
            else:
                identity_hits[tr.identity_label] += 1

            if writer is not None:
                label = tr.identity_label if tr.locked_identity else "desconocida"
                if label == "desconocida":
                    label = f"desconocida ID:{tr.track_id}"
                else:
                    label = f"{label} LOCK"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        for cur_bbox, cur_id in current_assignments:
            best_iou = 0.0
            best_prev_id = None
            for prev_bbox, prev_id in prev_assignments:
                iou = base05.calcular_iou(cur_bbox, prev_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_prev_id = prev_id
            if best_prev_id is not None and best_iou > 0.5 and best_prev_id != cur_id:
                id_switches += 1
        prev_assignments = current_assignments

        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    confirmed_tracks = [tr for tr in tracks.values() if tr.hits >= args.min_track_hits]
    locked_identities = sorted({tr.identity_label for tr in confirmed_tracks if tr.locked_identity})
    estimated_total = len(confirmed_tracks)
    count_error = estimated_total - args.expected_total_cows
    abs_error = abs(count_error)
    count_accuracy = max(0.0, 1.0 - abs_error / args.expected_total_cows) if args.expected_total_cows else 0.0

    report = {
        "video_in": args.video_in,
        "video_out": None if args.no_render else args.video_out,
        "processed_frames": processed,
        "start_frame": args.start_frame,
        "max_frames": args.max_frames,
        "expected_total_cows": args.expected_total_cows,
        "estimated_confirmed_tracks_alive": estimated_total,
        "count_error": count_error,
        "count_accuracy": count_accuracy,
        "mean_drawn_count_per_frame": float(np.mean(frame_counts)) if frame_counts else 0.0,
        "median_drawn_count_per_frame": float(np.median(frame_counts)) if frame_counts else 0.0,
        "max_drawn_count_per_frame": int(np.max(frame_counts)) if frame_counts else 0,
        "id_switches_approx": id_switches,
        "locked_identities": locked_identities,
        "identity_frame_hits": dict(sorted(identity_hits.items())),
        "unknown_frame_hits": unknown_hits,
        "params": vars(args),
    }
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("============================================================")
    print("Reporte Erondina estable")
    print("============================================================")
    print(f"Frames procesados             : {processed}")
    print(f"Tracks confirmados vivos       : {estimated_total}")
    print(f"Accuracy conteo estimado          : {100.0 * count_accuracy:.2f}%")
    print(f"Media cajas dibujadas/frame    : {report['mean_drawn_count_per_frame']:.2f}")
    print(f"ID switches aprox.             : {id_switches}")
    print(f"Identidades bloqueadas         : {', '.join(locked_identities) if locked_identities else 'ninguna'}")
    print(f"Reporte guardado en            : {args.report_out}")
    if not args.no_render:
        print(f"Video guardado en              : {args.video_out}")


if __name__ == "__main__":
    main()
