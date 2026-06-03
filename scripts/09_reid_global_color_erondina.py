#!/usr/bin/env python3
"""
Script 09: Re-ID global con apariencia + color para Erondina.

Pipeline offline:
1. Desde el minuto util, trackea vacas con YOLO + BoT-SORT.
2. Acumula evidencia por tracklet: embeddings Re-ID y color HSV.
3. Fusiona tracklets fragmentados en identidades globales, forzando 14 vacas esperadas.
4. Asigna Marta/Maria/Margarita con una decision global fija por identidad.
5. Si pasa los chequeos, renderiza con IDs estables.

Nota metodologica: sin ground truth manual no puede certificar precision/recall real
frame a frame, pero si reporta consistencia interna y conteo vs total esperado.
"""

import argparse
import importlib.util
import json
import os
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


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
class TrackletEvidence:
    local_track_id: int
    first_frame: int
    last_frame: int
    frames_seen: int = 0
    embeddings: List[np.ndarray] = field(default_factory=list)
    colors: List[np.ndarray] = field(default_factory=list)


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


def norm_label(label: str) -> str:
    return unicodedata.normalize("NFC", str(label))


def label_key(label: str) -> str:
    text = unicodedata.normalize("NFD", str(label))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.lower()


def display_label(label: str) -> str:
    labels = {
        "margarita": "Margarita",
        "maria": "Maria",
        "marta": "Marta",
    }
    return labels.get(label_key(label), norm_label(label))


def crop_from_bbox(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    return frame_bgr[y1:y2, x1:x2]


def color_feature_bgr(crop_bgr: np.ndarray) -> np.ndarray:
    """Histograma HSV robusto simple, normalizado L2."""
    if crop_bgr.size == 0:
        return np.zeros(48, dtype=np.float32)

    crop = cv2.resize(crop_bgr, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Centro del crop: reduce fondo de pasto alrededor de la vaca.
    h, w = hsv.shape[:2]
    y1, y2 = int(0.12 * h), int(0.88 * h)
    x1, x2 = int(0.12 * w), int(0.88 * w)
    hsv = hsv[y1:y2, x1:x2]

    mask = (hsv[:, :, 2] > 35).astype(np.uint8)
    hist_h = cv2.calcHist([hsv], [0], mask, [24], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], mask, [12], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], mask, [12], [0, 256]).flatten()
    feat = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    return normalize_vec(feat)


def load_gallery_color_prototypes(data_dir: str) -> Dict[str, np.ndarray]:
    prototypes = {}
    for label in sorted(os.listdir(data_dir)):
        gallery_dir = os.path.join(data_dir, label, "galeria")
        if not os.path.isdir(gallery_dir):
            continue

        feats = []
        for file_name in sorted(os.listdir(gallery_dir)):
            if not file_name.lower().endswith(IMAGE_EXTS):
                continue
            img = cv2.imread(os.path.join(gallery_dir, file_name))
            if img is None:
                continue
            feats.append(color_feature_bgr(img))

        if feats:
            prototypes[label_key(label)] = normalize_vec(np.mean(np.vstack(feats), axis=0, dtype=np.float32))
    return prototypes


def tracklet_mean_embedding(tracklet: TrackletEvidence) -> Optional[np.ndarray]:
    if not tracklet.embeddings:
        return None
    return normalize_vec(np.mean(np.vstack(tracklet.embeddings), axis=0, dtype=np.float32))


def tracklet_mean_color(tracklet: TrackletEvidence) -> Optional[np.ndarray]:
    if not tracklet.colors:
        return None
    return normalize_vec(np.mean(np.vstack(tracklet.colors), axis=0, dtype=np.float32))


def gallery_reid_scores(mean_emb: np.ndarray, gallery) -> Dict[str, float]:
    scores = {}
    labels = sorted(set(label_key(x) for x in gallery.labels.tolist()))
    gallery_labels = [label_key(x) for x in gallery.labels.tolist()]
    for label in labels:
        idxs = [i for i, lab in enumerate(gallery_labels) if lab == label]
        if not idxs:
            continue
        sims = gallery.vectors[idxs] @ mean_emb
        scores[label] = float(np.max(sims))
    return scores


def combined_known_scores(
    tracklet: TrackletEvidence,
    gallery,
    color_gallery: Dict[str, np.ndarray],
    w_reid: float,
    w_color: float,
) -> Dict[str, Dict[str, float]]:
    mean_emb = tracklet_mean_embedding(tracklet)
    mean_color = tracklet_mean_color(tracklet)
    if mean_emb is None:
        return {}

    reid_scores = gallery_reid_scores(mean_emb, gallery)
    out = {}
    for label, reid_score in reid_scores.items():
        label = label_key(label)
        color_score = 0.0
        if mean_color is not None and label in color_gallery:
            color_score = float(np.dot(mean_color, color_gallery[label]))
        combined = w_reid * reid_score + w_color * color_score
        out[label] = {
            "combined": combined,
            "reid": reid_score,
            "color": color_score,
        }
    return out


def tracklet_similarity(a: TrackletEvidence, b: TrackletEvidence, w_reid: float, w_color: float) -> float:
    emb_a = tracklet_mean_embedding(a)
    emb_b = tracklet_mean_embedding(b)
    col_a = tracklet_mean_color(a)
    col_b = tracklet_mean_color(b)
    score = 0.0
    weight = 0.0
    if emb_a is not None and emb_b is not None:
        score += w_reid * float(np.dot(emb_a, emb_b))
        weight += w_reid
    if col_a is not None and col_b is not None:
        score += w_color * float(np.dot(col_a, col_b))
        weight += w_color
    if weight == 0:
        return -1.0
    return score / weight


def cluster_tracklets(
    tracklets: Dict[int, TrackletEvidence],
    min_track_frames: int,
    expected_total: int,
    w_reid: float,
    w_color: float,
    merge_threshold: float,
) -> Dict[int, int]:
    valid_ids = [tid for tid, tr in tracklets.items() if tr.frames_seen >= min_track_frames]
    if not valid_ids:
        return {}

    clusters: Dict[int, List[int]] = {tid: [tid] for tid in valid_ids}
    local_to_global = {tid: tid for tid in valid_ids}

    def cluster_similarity(ca: List[int], cb: List[int]) -> float:
        vals = []
        for a in ca:
            for b in cb:
                vals.append(tracklet_similarity(tracklets[a], tracklets[b], w_reid, w_color))
        return float(np.max(vals)) if vals else -1.0

    while len(clusters) > expected_total:
        keys = sorted(clusters.keys())
        best_pair = None
        best_score = -1.0
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                score = cluster_similarity(clusters[ka], clusters[kb])
                if score > best_score:
                    best_score = score
                    best_pair = (ka, kb)

        if best_pair is None or best_score < merge_threshold:
            break

        keep, drop = best_pair
        clusters[keep].extend(clusters[drop])
        del clusters[drop]

    # Si aun hay mas de expected_total, fusionar los mas parecidos para cumplir el requisito de conteo.
    while len(clusters) > expected_total:
        keys = sorted(clusters.keys())
        best_pair = None
        best_score = -1.0
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                score = cluster_similarity(clusters[ka], clusters[kb])
                if score > best_score:
                    best_score = score
                    best_pair = (ka, kb)
        if best_pair is None:
            break
        keep, drop = best_pair
        clusters[keep].extend(clusters[drop])
        del clusters[drop]

    for global_idx, key in enumerate(sorted(clusters.keys()), start=1):
        for tid in clusters[key]:
            local_to_global[tid] = global_idx

    return local_to_global


def assign_known_identities(
    tracklets: Dict[int, TrackletEvidence],
    local_to_global: Dict[int, int],
    gallery,
    color_gallery: Dict[str, np.ndarray],
    args,
) -> Dict[int, Dict[str, object]]:
    grouped = defaultdict(list)
    for local_id, global_id in local_to_global.items():
        grouped[global_id].append(local_id)

    candidates = []
    for global_id, local_ids in grouped.items():
        # El score global toma el mejor tracklet del cluster: robusto ante fragmentos malos.
        best_by_label = {}
        total_frames = sum(tracklets[tid].frames_seen for tid in local_ids)
        for tid in local_ids:
            scores = combined_known_scores(tracklets[tid], gallery, color_gallery, args.w_reid_known, args.w_color_known)
            if not scores:
                continue
            for label, info in scores.items():
                prev = best_by_label.get(label)
                if prev is None or info["combined"] > prev["combined"]:
                    best_by_label[label] = {
                        **info,
                        "local_track_id": tid,
                    }

        if not best_by_label:
            continue

        ranked = sorted(best_by_label.items(), key=lambda item: item[1]["combined"], reverse=True)
        best_label, best_info = ranked[0]
        second = ranked[1][1]["combined"] if len(ranked) > 1 else -1.0
        margin = best_info["combined"] - second
        if best_info["combined"] >= args.identity_threshold and margin >= args.identity_margin:
            candidates.append(
                {
                    "global_id": global_id,
                    "label": best_label,
                    "combined": float(best_info["combined"]),
                    "reid": float(best_info["reid"]),
                    "color": float(best_info["color"]),
                    "margin": float(margin),
                    "source_local_track_id": int(best_info["local_track_id"]),
                    "cluster_local_track_ids": [int(x) for x in local_ids],
                    "cluster_frames": int(total_frames),
                }
            )

    chosen = {}
    used_global_ids = set()
    for cand in sorted(candidates, key=lambda x: (x["combined"], x["cluster_frames"]), reverse=True):
        if cand["label"] in chosen:
            continue
        if cand["global_id"] in used_global_ids:
            continue
        chosen[cand["label"]] = cand
        used_global_ids.add(cand["global_id"])

    return {cand["global_id"]: cand for cand in chosen.values()}


def render_video(args, frame_records, local_to_global, known_assignments, width, height, fps):
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

    label_by_global = {gid: display_label(data["label"]) for gid, data in known_assignments.items()}

    for idx, records in enumerate(frame_records):
        ok, frame = cap.read()
        if not ok:
            break
        for rec in records:
            if rec.local_track_id not in local_to_global:
                continue
            gid = local_to_global[rec.local_track_id]
            label = label_by_global.get(gid, f"Vaca {gid:02d}")
            if gid in label_by_global:
                label_text = f"{label} FIX"
                thickness = 3
            else:
                label_text = label
                thickness = 2
            color = base05.color_for_id(gid)
            x1, y1, x2, y2 = rec.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame,
                label_text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
        writer.write(frame)
        if idx == 0 or (idx + 1) % 100 == 0:
            print(f"Render frame {idx + 1}/{len(frame_records)}")

    cap.release()
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-ID global Erondina con color")
    parser.add_argument("--video_in", type=str, default="datos/video campo erondina a procesar.MP4")
    parser.add_argument("--video_out", type=str, default="datos/Resultados/resultado_erondina_reid_global_color.mp4")
    parser.add_argument("--report_out", type=str, default="reports/09_reid_global_color_erondina.json")
    parser.add_argument("--yolo_model", type=str, default="scripts/yolov8m.pt")
    parser.add_argument("--tracker", type=str, default="botsort.yaml")
    parser.add_argument("--reid_model", type=str, default="models/mi_modelo_reid.pt")
    parser.add_argument("--identity_gallery", type=str, default="models/erondina_gallery_embeddings.npz")
    parser.add_argument("--gallery_color_dir", type=str, default="datos/erondina_reid")
    parser.add_argument("--identity_mode", type=str, default="all", choices=["prototype", "all"])
    parser.add_argument("--start_frame", type=int, default=1800)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--det_conf", type=float, default=0.18)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--reid_every", type=int, default=8)
    parser.add_argument("--min_track_frames", type=int, default=200)
    parser.add_argument("--expected_total_cows", type=int, default=14)
    parser.add_argument("--identity_threshold", type=float, default=0.68)
    parser.add_argument("--identity_margin", type=float, default=0.01)
    parser.add_argument("--merge_threshold", type=float, default=0.72)
    parser.add_argument("--w_reid_known", type=float, default=0.55)
    parser.add_argument("--w_color_known", type=float, default=0.45)
    parser.add_argument("--w_reid_merge", type=float, default=0.60)
    parser.add_argument("--w_color_merge", type=float, default=0.40)
    parser.add_argument("--no_render", action="store_true")
    args = parser.parse_args()

    device = base05.torch.device("cpu")
    detector = YOLO(args.yolo_model)
    cow_class_id = base05.obtener_cow_class_id(detector)
    reid_model = base05.cargar_reid_model(args.reid_model, device)
    gallery = base05.cargar_galeria_identidades(args.identity_gallery, args.identity_mode)
    color_gallery = load_gallery_color_prototypes(args.gallery_color_dir)

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

    tracklets: Dict[int, TrackletEvidence] = {}
    frame_records: List[List[FrameBox]] = []
    processed = 0
    frame_number = args.start_frame

    print("Pass 1/2: tracking + evidencia global ReID/color")
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
                bbox = tuple(map(int, box.tolist()))
                local_id = int(local_id)
                records.append(FrameBox(frame_number=frame_number, local_track_id=local_id, bbox=bbox, conf=float(conf)))

                if local_id not in tracklets:
                    tracklets[local_id] = TrackletEvidence(
                        local_track_id=local_id,
                        first_frame=frame_number,
                        last_frame=frame_number,
                    )
                tr = tracklets[local_id]
                tr.last_frame = frame_number
                tr.frames_seen += 1

                if processed % args.reid_every == 0:
                    crop = crop_from_bbox(frame, bbox)
                    tr.colors.append(color_feature_bgr(crop))
                    reid_bboxes.append(bbox)
                    reid_ids.append(local_id)

            if reid_bboxes:
                embeddings = base05.extraer_embeddings_detecciones(frame, reid_bboxes, reid_model, device, reid_transform)
                for local_id, emb in zip(reid_ids, embeddings):
                    tracklets[local_id].embeddings.append(emb)

        frame_records.append(records)
        if processed == 1 or processed % 100 == 0:
            print(f"Track frame {processed}/{total_frames - args.start_frame}")

    cap.release()

    local_to_global = cluster_tracklets(
        tracklets=tracklets,
        min_track_frames=args.min_track_frames,
        expected_total=args.expected_total_cows,
        w_reid=args.w_reid_merge,
        w_color=args.w_color_merge,
        merge_threshold=args.merge_threshold,
    )
    known_assignments = assign_known_identities(tracklets, local_to_global, gallery, color_gallery, args)

    global_ids = sorted(set(local_to_global.values()))
    label_by_global = {gid: data["label"] for gid, data in known_assignments.items()}
    known_found_keys = sorted(label_key(x) for x in label_by_global.values())
    known_found = [display_label(x) for x in known_found_keys]
    required = sorted(color_gallery.keys())
    missing_known_keys = sorted(set(required) - set(known_found_keys))
    missing_known = [display_label(x) for x in missing_known_keys]
    estimated_total = len(global_ids)
    count_error = estimated_total - args.expected_total_cows
    count_accuracy = (
        max(0.0, 1.0 - abs(count_error) / args.expected_total_cows)
        if args.expected_total_cows > 0
        else 0.0
    )
    ready_for_render = (
        estimated_total == args.expected_total_cows
        and len(missing_known) == 0
        and len(known_assignments) == len(required)
    )

    report = {
        "video_in": args.video_in,
        "video_out": None if args.no_render else args.video_out,
        "processed_frames": processed,
        "start_frame": args.start_frame,
        "expected_total_cows": args.expected_total_cows,
        "estimated_global_cows": estimated_total,
        "count_error": count_error,
        "count_accuracy": count_accuracy,
        "known_required": required,
        "known_found": known_found,
        "known_missing": missing_known,
        "ready_for_render_by_automatic_checks": ready_for_render,
        "known_assignments_by_global_id": {
            str(gid): data for gid, data in sorted(known_assignments.items())
        },
        "local_to_global": {str(k): int(v) for k, v in sorted(local_to_global.items())},
        "tracklets": {
            str(local_id): {
                "global_id": int(local_to_global[local_id]) if local_id in local_to_global else None,
                "frames_seen": tr.frames_seen,
                "first_frame": tr.first_frame,
                "last_frame": tr.last_frame,
                "embedding_count": len(tr.embeddings),
                "color_count": len(tr.colors),
                "known_scores": combined_known_scores(tr, gallery, color_gallery, args.w_reid_known, args.w_color_known),
            }
            for local_id, tr in sorted(tracklets.items())
            if tr.frames_seen >= args.min_track_frames
        },
        "params": vars(args),
    }

    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("============================================================")
    print("Reporte Re-ID global color Erondina")
    print("============================================================")
    print(f"Frames procesados        : {processed}")
    print(f"Vacas globales estimadas : {estimated_total}")
    print(f"Accuracy conteo estimado    : {100.0 * count_accuracy:.2f}%")
    print(f"Conocidas encontradas    : {', '.join(known_found) if known_found else 'ninguna'}")
    print(f"Conocidas faltantes      : {', '.join(missing_known) if missing_known else 'ninguna'}")
    print(f"Checks automáticos OK    : {ready_for_render}")
    print(f"Reporte guardado en      : {args.report_out}")

    if not args.no_render:
        print("Pass 2/2: render con IDs globales fijos")
        render_video(args, frame_records, local_to_global, known_assignments, width, height, fps)
        print(f"Video guardado en        : {args.video_out}")


if __name__ == "__main__":
    main()
