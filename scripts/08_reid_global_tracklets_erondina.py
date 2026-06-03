#!/usr/bin/env python3
"""
Script 08: Re-ID global por trayectorias completas.

Este enfoque evita cambiar identidades frame a frame:
1. Recorre el video desde el minuto util y crea tracklets con BoT-SORT/ByteTrack.
2. Acumula embeddings Re-ID por track durante todo el video.
3. Decide una unica identidad fija por track usando evidencia global.
4. Renderiza el video con la identidad ya congelada.

Si Marta/Maria/Margarita no superan el umbral global, quedan como desconocidas.
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
    track_id: int
    bbox: Tuple[int, int, int, int]
    conf: float


@dataclass
class TrackletEvidence:
    track_id: int
    first_frame: int
    last_frame: int
    frames_seen: int = 0
    embeddings: List[np.ndarray] = field(default_factory=list)
    label_votes: Counter = field(default_factory=Counter)
    label_scores: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))


def predict_identity(embedding: np.ndarray, gallery, threshold: float) -> Tuple[str, float]:
    label, score = base05.reconocer_identidad(embedding, gallery, threshold)
    return label, score


def identity_scores_for_track(tracklet: TrackletEvidence, gallery) -> Dict[str, float]:
    if not tracklet.embeddings:
        return {}

    mat = np.vstack(tracklet.embeddings).astype(np.float32)
    mean_emb = np.mean(mat, axis=0, dtype=np.float32)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

    scores = {}
    labels = sorted(set(str(x) for x in gallery.labels.tolist()))
    for label in labels:
        idxs = [i for i, lab in enumerate(gallery.labels.tolist()) if str(lab) == label]
        if not idxs:
            continue
        label_vectors = gallery.vectors[idxs]
        sims = label_vectors @ mean_emb
        scores[label] = float(np.max(sims))
    return scores


def assign_global_identities(
    tracklets: Dict[int, TrackletEvidence],
    gallery,
    min_track_frames: int,
    identity_threshold: float,
    identity_margin: float,
) -> Dict[int, Dict[str, object]]:
    candidates = []

    for track_id, tracklet in tracklets.items():
        if tracklet.frames_seen < min_track_frames:
            continue

        scores = identity_scores_for_track(tracklet, gallery)
        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_label, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = best_score - second_score

        if best_score >= identity_threshold and margin >= identity_margin:
            candidates.append(
                {
                    "track_id": track_id,
                    "label": best_label,
                    "score": best_score,
                    "margin": margin,
                    "frames_seen": tracklet.frames_seen,
                    "first_frame": tracklet.first_frame,
                    "last_frame": tracklet.last_frame,
                    "all_scores": scores,
                }
            )

    # Una identidad conocida solo puede pertenecer a un track.
    chosen_by_label = {}
    for cand in sorted(candidates, key=lambda x: (x["score"], x["frames_seen"]), reverse=True):
        label = cand["label"]
        if label in chosen_by_label:
            continue
        chosen_by_label[label] = cand

    assignments = {}
    for cand in chosen_by_label.values():
        assignments[cand["track_id"]] = cand
    return assignments


def render_video(
    args,
    frame_records: List[List[FrameBox]],
    assignments: Dict[int, Dict[str, object]],
    width: int,
    height: int,
    fps: float,
) -> None:
    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {args.video_in}")
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    os.makedirs(os.path.dirname(args.video_out), exist_ok=True)
    writer = cv2.VideoWriter(
        args.video_out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        (width, height),
    )

    processed = 0
    while processed < len(frame_records):
        ok, frame = cap.read()
        if not ok:
            break

        for item in frame_records[processed]:
            x1, y1, x2, y2 = item.bbox
            color = base05.color_for_id(item.track_id)
            if item.track_id in assignments:
                label = str(assignments[item.track_id]["label"])
                text = f"{label} FIX"
                thickness = 3
            else:
                text = f"desconocida ID:{item.track_id}"
                thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        processed += 1
        if processed == 1 or processed % 100 == 0:
            print(f"Render frame {processed}/{len(frame_records)}")

    cap.release()
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-ID global por tracklets completos en Erondina")
    parser.add_argument("--video_in", type=str, default="datos/video campo erondina a procesar.MP4")
    parser.add_argument("--video_out", type=str, default="datos/Resultados/resultado_erondina_reid_global.mp4")
    parser.add_argument("--report_out", type=str, default="reports/08_reid_global_tracklets_erondina.json")
    parser.add_argument("--yolo_model", type=str, default="scripts/yolov8m.pt")
    parser.add_argument("--tracker", type=str, default="botsort.yaml")
    parser.add_argument("--reid_model", type=str, default="models/mi_modelo_reid.pt")
    parser.add_argument("--identity_gallery", type=str, default="models/erondina_gallery_embeddings.npz")
    parser.add_argument("--identity_mode", type=str, default="prototype", choices=["prototype", "all"])
    parser.add_argument("--start_frame", type=int, default=1800)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--det_conf", type=float, default=0.18)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--reid_every", type=int, default=8)
    parser.add_argument("--min_track_frames", type=int, default=45)
    parser.add_argument("--identity_threshold", type=float, default=0.72)
    parser.add_argument("--identity_margin", type=float, default=0.02)
    parser.add_argument("--expected_total_cows", type=int, default=14)
    parser.add_argument("--no_render", action="store_true")
    args = parser.parse_args()

    device = base05.torch.device("cpu")
    detector = YOLO(args.yolo_model)
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

    frame_records: List[List[FrameBox]] = []
    tracklets: Dict[int, TrackletEvidence] = {}
    processed = 0
    frame_number = args.start_frame

    print("Pass 1/2: tracking + evidencia ReID global")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_number += 1
        processed += 1
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
            classes=[base05.obtener_cow_class_id(detector)],
        )[0]

        records = []
        if result.boxes is not None and result.boxes.id is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            crops_for_reid = []
            reid_track_ids = []
            for box, track_id, conf in zip(boxes, ids, confs):
                bbox = tuple(map(int, box.tolist()))
                records.append(FrameBox(frame_number=frame_number, track_id=int(track_id), bbox=bbox, conf=float(conf)))

                if track_id not in tracklets:
                    tracklets[int(track_id)] = TrackletEvidence(
                        track_id=int(track_id),
                        first_frame=frame_number,
                        last_frame=frame_number,
                    )
                tr = tracklets[int(track_id)]
                tr.last_frame = frame_number
                tr.frames_seen += 1

                if processed % args.reid_every == 0:
                    crops_for_reid.append(bbox)
                    reid_track_ids.append(int(track_id))

            if crops_for_reid:
                embeddings = base05.extraer_embeddings_detecciones(frame, crops_for_reid, reid_model, device, reid_transform)
                for track_id, emb in zip(reid_track_ids, embeddings):
                    tr = tracklets[track_id]
                    tr.embeddings.append(emb)
                    label, score = predict_identity(emb, gallery, args.identity_threshold)
                    if label != "desconocida":
                        tr.label_votes[label] += 1
                        tr.label_scores[label].append(score)

        frame_records.append(records)
        if processed == 1 or processed % 100 == 0:
            print(f"Track frame {processed}/{total_frames - args.start_frame}")

    cap.release()

    assignments = assign_global_identities(
        tracklets,
        gallery,
        min_track_frames=args.min_track_frames,
        identity_threshold=args.identity_threshold,
        identity_margin=args.identity_margin,
    )

    confirmed_track_ids = [
        track_id for track_id, tr in tracklets.items()
        if tr.frames_seen >= args.min_track_frames
    ]
    estimated_total = len(confirmed_track_ids)
    count_error = estimated_total - args.expected_total_cows
    count_accuracy = (
        max(0.0, 1.0 - abs(count_error) / args.expected_total_cows)
        if args.expected_total_cows > 0
        else 0.0
    )

    report = {
        "video_in": args.video_in,
        "video_out": None if args.no_render else args.video_out,
        "processed_frames": processed,
        "start_frame": args.start_frame,
        "expected_total_cows": args.expected_total_cows,
        "confirmed_track_count": estimated_total,
        "count_error": count_error,
        "count_accuracy": count_accuracy,
        "assigned_identities": {
            str(track_id): assignment for track_id, assignment in sorted(assignments.items())
        },
        "tracklets": {
            str(track_id): {
                "frames_seen": tr.frames_seen,
                "first_frame": tr.first_frame,
                "last_frame": tr.last_frame,
                "embedding_count": len(tr.embeddings),
                "votes": dict(tr.label_votes),
                "global_scores": identity_scores_for_track(tr, gallery),
            }
            for track_id, tr in sorted(tracklets.items())
            if tr.frames_seen >= args.min_track_frames
        },
        "params": vars(args),
        "note": (
            "Las identidades se asignan una sola vez por trayectoria completa. "
            "El render no vuelve a clasificar frame a frame."
        ),
    }

    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("============================================================")
    print("Reporte Re-ID global por tracklets")
    print("============================================================")
    print(f"Frames procesados          : {processed}")
    print(f"Tracks confirmados         : {estimated_total}")
    print(f"Accuracy conteo vs esperado: {100.0 * count_accuracy:.2f}%")
    print("Identidades asignadas:")
    for track_id, assignment in sorted(assignments.items()):
        print(
            f"  Track {track_id}: {assignment['label']} "
            f"score={assignment['score']:.3f} margin={assignment['margin']:.3f}"
        )
    print(f"Reporte guardado en        : {args.report_out}")

    if not args.no_render:
        print("Pass 2/2: render con identidades fijas")
        render_video(args, frame_records, assignments, width, height, fps)
        print(f"Video guardado en          : {args.video_out}")


if __name__ == "__main__":
    main()
