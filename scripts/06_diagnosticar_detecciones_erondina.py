#!/usr/bin/env python3
"""
Script 06: Diagnóstico rápido de detecciones en el video Erondina.

Muestrea frames del video y compara cuántas vacas detecta YOLO con distintos
umbrales de confianza. Sirve para calibrar antes de renderizar el video completo.
"""

import argparse
import json
import os
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def obtener_cow_class_id(yolo_model: YOLO) -> int:
    names = yolo_model.model.names
    for cls_id, cls_name in names.items():
        if str(cls_name).lower() == "cow":
            return int(cls_id)
    return 19


def calcular_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return float(inter_area / (area_a + area_b - inter_area + 1e-8))


def nms_manual(
    bboxes: List[Tuple[int, int, int, int]],
    confs: List[float],
    iou_thr: float,
) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    if not bboxes:
        return [], []

    order = sorted(range(len(bboxes)), key=lambda i: confs[i], reverse=True)
    keep = []
    for idx in order:
        if all(calcular_iou(bboxes[idx], bboxes[kept_idx]) <= iou_thr for kept_idx in keep):
            keep.append(idx)
    return [bboxes[i] for i in keep], [confs[i] for i in keep]


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


def detectar_frame(
    model: YOLO,
    frame,
    cow_class_id: int,
    conf_thr: float,
    imgsz: int,
    iou_thr: float,
    min_area_ratio: float,
    max_area_ratio: float,
    min_aspect: float,
    max_aspect: float,
):
    h, w = frame.shape[:2]
    result = model.predict(source=frame, conf=conf_thr, imgsz=imgsz, verbose=False, device="cpu")[0]
    bboxes = []
    confs = []
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        for box, cls_id, score in zip(xyxy, cls, scores):
            if cls_id != cow_class_id or score < conf_thr:
                continue
            bboxes.append(tuple(map(int, box.tolist())))
            confs.append(float(score))

    bboxes, confs = nms_manual(bboxes, confs, iou_thr=iou_thr)
    bboxes, confs = filtrar_geometria(
        bboxes,
        confs,
        frame_w=w,
        frame_h=h,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
    )
    return bboxes, confs


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico de detecciones YOLO en Erondina")
    parser.add_argument("--video_in", type=str, default="datos/video campo erondina a procesar.MP4")
    parser.add_argument("--yolo_model", type=str, default="scripts/yolov8m.pt")
    parser.add_argument("--report_out", type=str, default="reports/06_diagnostico_detecciones_erondina.json")
    parser.add_argument("--expected_total_cows", type=int, default=14)
    parser.add_argument("--sample_step", type=int, default=500)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--thresholds", type=str, default="0.15,0.25,0.35,0.45,0.55,0.65")
    parser.add_argument("--iou_threshold", type=float, default=0.70)
    parser.add_argument("--min_area_ratio", type=float, default=0.00015)
    parser.add_argument("--max_area_ratio", type=float, default=0.02000)
    parser.add_argument("--min_aspect", type=float, default=0.35)
    parser.add_argument("--max_aspect", type=float, default=3.50)
    args = parser.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    model = YOLO(args.yolo_model)
    cow_class_id = obtener_cow_class_id(model)
    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {args.video_in}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_frames = list(range(0, total_frames, args.sample_step))
    rows = []
    summary = {}

    for thr in thresholds:
        counts = []
        for frame_idx in sampled_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            bboxes, confs = detectar_frame(
                model=model,
                frame=frame,
                cow_class_id=cow_class_id,
                conf_thr=thr,
                imgsz=args.imgsz,
                iou_thr=args.iou_threshold,
                min_area_ratio=args.min_area_ratio,
                max_area_ratio=args.max_area_ratio,
                min_aspect=args.min_aspect,
                max_aspect=args.max_aspect,
            )
            count = len(bboxes)
            counts.append(count)
            rows.append({"frame": frame_idx, "threshold": thr, "count": count})

        if counts:
            errors = [abs(c - args.expected_total_cows) for c in counts]
            summary[str(thr)] = {
                "mean_count": float(np.mean(counts)),
                "median_count": float(np.median(counts)),
                "min_count": int(np.min(counts)),
                "max_count": int(np.max(counts)),
                "mean_abs_error_vs_expected": float(np.mean(errors)),
            }

    cap.release()
    report = {
        "video_in": args.video_in,
        "yolo_model": args.yolo_model,
        "expected_total_cows": args.expected_total_cows,
        "sample_step": args.sample_step,
        "sampled_frames": sampled_frames,
        "filters": {
            "iou_threshold": args.iou_threshold,
            "min_area_ratio": args.min_area_ratio,
            "max_area_ratio": args.max_area_ratio,
            "min_aspect": args.min_aspect,
            "max_aspect": args.max_aspect,
        },
        "summary": summary,
        "rows": rows,
    }

    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("============================================================")
    print("Diagnóstico detecciones Erondina")
    print("============================================================")
    for thr, stats in summary.items():
        print(
            f"thr={thr:>4} | mean={stats['mean_count']:5.2f} | "
            f"median={stats['median_count']:5.2f} | min={stats['min_count']:2d} | "
            f"max={stats['max_count']:2d} | MAE vs 14={stats['mean_abs_error_vs_expected']:5.2f}"
        )
    print(f"Reporte guardado en: {args.report_out}")


if __name__ == "__main__":
    main()
