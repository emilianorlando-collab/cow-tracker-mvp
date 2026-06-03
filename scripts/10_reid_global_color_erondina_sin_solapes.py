#!/usr/bin/env python3
"""
Script 10: Re-ID global con color, sin fusionar trayectorias solapadas.

Corrige el problema observado en el Script 09: si dos tracklets aparecen en el
mismo frame, no pueden representar la misma vaca. Por eso esta version:
- mantiene IDs conocidos unicos;
- no fusiona tracklets con solape temporal significativo;
- genera un reporte y una planilla visual de candidatos antes de renderizar;
- solo renderiza si se lo pide explicitamente.
"""

import argparse
import importlib.util
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def cargar_modulo(nombre: str, path: str):
    spec = importlib.util.spec_from_file_location(nombre, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base09 = cargar_modulo("base09", os.path.join(SCRIPT_DIR, "09_reid_global_color_erondina.py"))


def interval_overlap(a, b) -> int:
    return max(0, min(a.last_frame, b.last_frame) - max(a.first_frame, b.first_frame) + 1)


def can_merge(tracklets, cluster_a: List[int], cluster_b: List[int], max_overlap_frames: int) -> bool:
    for a in cluster_a:
        for b in cluster_b:
            if interval_overlap(tracklets[a], tracklets[b]) > max_overlap_frames:
                return False
    return True


def cluster_tracklets_no_overlap(
    tracklets,
    min_track_frames: int,
    expected_total: int,
    w_reid: float,
    w_color: float,
    merge_threshold: float,
    max_overlap_frames: int,
) -> Dict[int, int]:
    valid_ids = [tid for tid, tr in tracklets.items() if tr.frames_seen >= min_track_frames]
    clusters: Dict[int, List[int]] = {tid: [tid] for tid in valid_ids}
    if not clusters:
        return {}

    def cluster_similarity(ca: List[int], cb: List[int]) -> float:
        vals = []
        for a in ca:
            for b in cb:
                vals.append(base09.tracklet_similarity(tracklets[a], tracklets[b], w_reid, w_color))
        return float(np.max(vals)) if vals else -1.0

    while True:
        keys = sorted(clusters.keys())
        best_pair = None
        best_score = -1.0
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                if not can_merge(tracklets, clusters[ka], clusters[kb], max_overlap_frames):
                    continue
                score = cluster_similarity(clusters[ka], clusters[kb])
                if score > best_score:
                    best_score = score
                    best_pair = (ka, kb)

        if best_pair is None or best_score < merge_threshold:
            break

        keep, drop = best_pair
        clusters[keep].extend(clusters[drop])
        del clusters[drop]

    local_to_global = {}
    for global_idx, key in enumerate(sorted(clusters.keys()), start=1):
        for tid in clusters[key]:
            local_to_global[tid] = global_idx

    return local_to_global


def group_tracklets(local_to_global: Dict[int, int]) -> Dict[int, List[int]]:
    grouped = defaultdict(list)
    for local_id, global_id in local_to_global.items():
        grouped[global_id].append(local_id)
    return dict(grouped)


def validate_no_known_duplicates(frame_records, local_to_global, known_assignments) -> Dict[str, object]:
    label_by_gid = {gid: data["label"] for gid, data in known_assignments.items()}
    duplicate_frames = defaultdict(int)

    for records in frame_records:
        labels_this_frame = defaultdict(int)
        for rec in records:
            gid = local_to_global.get(rec.local_track_id)
            if gid is None or gid not in label_by_gid:
                continue
            labels_this_frame[label_by_gid[gid]] += 1
        for label, count in labels_this_frame.items():
            if count > 1:
                duplicate_frames[label] += 1

    return {
        "duplicate_known_label_frames": dict(duplicate_frames),
        "has_duplicate_known_labels": any(v > 0 for v in duplicate_frames.values()),
    }


def extract_track_crop(video_in: str, frame_number: int, bbox: Tuple[int, int, int, int]):
    cap = cv2.VideoCapture(video_in)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return np.zeros((180, 220, 3), dtype=np.uint8)
    crop = base09.crop_from_bbox(frame, bbox)
    if crop.size == 0:
        return np.zeros((180, 220, 3), dtype=np.uint8)
    return crop


def make_contact_sheet(args, frame_records, tracklets, local_to_global, known_assignments, out_path: str):
    rows = []
    gid_to_label = {gid: base09.display_label(data["label"]) for gid, data in known_assignments.items()}
    grouped = group_tracklets(local_to_global)

    for gid, label in sorted(gid_to_label.items(), key=lambda item: item[1]):
        local_ids = grouped.get(gid, [])
        candidate_records = []
        for records in frame_records:
            for rec in records:
                if rec.local_track_id in local_ids:
                    candidate_records.append(rec)
        if not candidate_records:
            continue
        picks = [
            candidate_records[0],
            candidate_records[len(candidate_records) // 2],
            candidate_records[-1],
        ]
        cells = []
        for rec in picks:
            crop = extract_track_crop(args.video_in, rec.frame_number, rec.bbox)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-ID global sin solapes temporales")
    parser.add_argument("--video_in", type=str, default="datos/video campo erondina a procesar.MP4")
    parser.add_argument("--video_out", type=str, default="datos/Resultados/resultado_erondina_reid_global_color_sin_solapes.mp4")
    parser.add_argument("--report_out", type=str, default="reports/10_reid_global_color_sin_solapes.json")
    parser.add_argument("--contact_sheet_out", type=str, default="reports/10_contact_sheet_conocidas.jpg")
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
    parser.add_argument("--max_overlap_frames", type=int, default=5)
    parser.add_argument("--w_reid_known", type=float, default=0.55)
    parser.add_argument("--w_color_known", type=float, default=0.45)
    parser.add_argument("--w_reid_merge", type=float, default=0.60)
    parser.add_argument("--w_color_merge", type=float, default=0.40)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # Reutiliza la pasada de tracking del script 09, pero interponemos el clustering sin solapes.
    device = base09.base05.torch.device("cpu")
    detector = base09.YOLO(args.yolo_model)
    cow_class_id = base09.base05.obtener_cow_class_id(detector)
    reid_model = base09.base05.cargar_reid_model(args.reid_model, device)
    gallery = base09.base05.cargar_galeria_identidades(args.identity_gallery, args.identity_mode)
    color_gallery = base09.load_gallery_color_prototypes(args.gallery_color_dir)
    reid_transform = base09.base05.transforms.Compose(
        [
            base09.base05.transforms.ToPILImage(),
            base09.base05.transforms.Resize((224, 224)),
            base09.base05.transforms.ToTensor(),
            base09.base05.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    tracklets = {}
    frame_records = []
    processed = 0
    frame_number = args.start_frame

    print("Pass 1/2: tracking + evidencia global ReID/color sin solapes")
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
                records.append(base09.FrameBox(frame_number=frame_number, local_track_id=local_id, bbox=bbox, conf=float(conf)))

                if local_id not in tracklets:
                    tracklets[local_id] = base09.TrackletEvidence(
                        local_track_id=local_id,
                        first_frame=frame_number,
                        last_frame=frame_number,
                    )
                tr = tracklets[local_id]
                tr.last_frame = frame_number
                tr.frames_seen += 1

                if processed % args.reid_every == 0:
                    crop = base09.crop_from_bbox(frame, bbox)
                    tr.colors.append(base09.color_feature_bgr(crop))
                    reid_bboxes.append(bbox)
                    reid_ids.append(local_id)

            if reid_bboxes:
                embeddings = base09.base05.extraer_embeddings_detecciones(frame, reid_bboxes, reid_model, device, reid_transform)
                for local_id, emb in zip(reid_ids, embeddings):
                    tracklets[local_id].embeddings.append(emb)

        frame_records.append(records)
        if processed == 1 or processed % 100 == 0:
            print(f"Track frame {processed}/{total_frames - args.start_frame}")

    cap.release()

    local_to_global = cluster_tracklets_no_overlap(
        tracklets=tracklets,
        min_track_frames=args.min_track_frames,
        expected_total=args.expected_total_cows,
        w_reid=args.w_reid_merge,
        w_color=args.w_color_merge,
        merge_threshold=args.merge_threshold,
        max_overlap_frames=args.max_overlap_frames,
    )
    known_assignments = base09.assign_known_identities(tracklets, local_to_global, gallery, color_gallery, args)
    duplicate_report = validate_no_known_duplicates(frame_records, local_to_global, known_assignments)
    contact_sheet = make_contact_sheet(args, frame_records, tracklets, local_to_global, known_assignments, args.contact_sheet_out)

    global_ids = sorted(set(local_to_global.values()))
    known_found = sorted(base09.display_label(data["label"]) for data in known_assignments.values())
    required_keys = sorted(color_gallery.keys())
    found_keys = sorted(base09.label_key(data["label"]) for data in known_assignments.values())
    missing_known = [base09.display_label(k) for k in sorted(set(required_keys) - set(found_keys))]
    estimated_total = len(global_ids)
    count_error = estimated_total - args.expected_total_cows
    count_accuracy = max(0.0, 1.0 - abs(count_error) / args.expected_total_cows) if args.expected_total_cows else 0.0
    ready_for_render = (
        len(missing_known) == 0
        and not duplicate_report["has_duplicate_known_labels"]
        and estimated_total >= args.expected_total_cows
    )

    report = {
        "video_in": args.video_in,
        "video_out": args.video_out if args.render else None,
        "processed_frames": processed,
        "expected_total_cows": args.expected_total_cows,
        "estimated_global_cows": estimated_total,
        "count_error": count_error,
        "count_accuracy": count_accuracy,
        "known_found": known_found,
        "known_missing": missing_known,
        "duplicate_report": duplicate_report,
        "ready_for_render_by_automatic_checks": ready_for_render,
        "known_assignments_by_global_id": {str(k): v for k, v in sorted(known_assignments.items())},
        "local_to_global": {str(k): int(v) for k, v in sorted(local_to_global.items())},
        "contact_sheet": contact_sheet,
        "params": vars(args),
    }

    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("============================================================")
    print("Reporte Re-ID global sin solapes")
    print("============================================================")
    print(f"Frames procesados        : {processed}")
    print(f"Vacas globales estimadas : {estimated_total}")
    print(f"Accuracy conteo estimado    : {100.0 * count_accuracy:.2f}%")
    print(f"Conocidas encontradas    : {', '.join(known_found) if known_found else 'ninguna'}")
    print(f"Conocidas faltantes      : {', '.join(missing_known) if missing_known else 'ninguna'}")
    print(f"Duplicados conocidos     : {duplicate_report['duplicate_known_label_frames']}")
    print(f"Checks automáticos OK    : {ready_for_render}")
    print(f"Reporte guardado en      : {args.report_out}")
    if contact_sheet:
        print(f"Contacto visual guardado : {contact_sheet}")

    if args.render:
        print("Pass 2/2: render con IDs globales sin solapes")
        base09.render_video(args, frame_records, local_to_global, known_assignments, width, height, fps)
        print(f"Video guardado en        : {args.video_out}")


if __name__ == "__main__":
    main()
