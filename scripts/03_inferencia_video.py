import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from ultralytics import YOLO


@dataclass
class TrackState:
    track_id: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    embedding: np.ndarray
    last_seen: int
    missed: int = 0


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


def cargar_reid_model(model_path: str, device: torch.device) -> nn.Module:
    model = ReIDFeatureExtractor().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def obtener_cow_class_id(yolo_model: YOLO) -> int:
    """
    Busca el ID de la clase 'cow' en los nombres del modelo COCO.
    Si no la encuentra, usa fallback 21 (común en COCO).
    """
    names = yolo_model.model.names
    for cls_id, cls_name in names.items():
        if str(cls_name).lower() == "cow":
            return int(cls_id)
    return 21


def recortar_bbox(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    return frame_bgr[y1:y2, x1:x2]


def bbox_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    # Color determinista por ID para mantener consistencia visual en todo el video.
    rng = np.random.default_rng(seed=track_id * 1337)
    c = rng.integers(low=40, high=255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


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
    union = area_a + area_b - inter_area + 1e-8

    return float(inter_area / union)


def calcular_iomin(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Intersección / área mínima entre dos cajas (IoMin)."""
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
    min_area = max(1.0, float(min(area_a, area_b)))
    return float(inter_area / min_area)


def filtrar_anti_mega_cajas(
    bboxes: List[Tuple[int, int, int, int]],
    confs: List[float],
    frame_w: int,
    frame_h: int,
    iomin_thr: float = 0.80,
    max_side_ratio: float = 0.25,
) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    """
    Elimina mega-cajas que envuelven otras vacas:
    1) Descarta cajas absurdamente grandes (ancho o alto > 25% del frame).
    2) Si IoMin > 0.80 entre dos cajas, descarta la caja más grande.
    """
    if not bboxes:
        return [], []

    keep = [True] * len(bboxes)

    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        if bw > max_side_ratio * frame_w or bh > max_side_ratio * frame_h:
            keep[i] = False

    for i in range(len(bboxes)):
        if not keep[i]:
            continue
        xi1, yi1, xi2, yi2 = bboxes[i]
        area_i = max(1, (xi2 - xi1) * (yi2 - yi1))
        for j in range(i + 1, len(bboxes)):
            if not keep[j]:
                continue
            xj1, yj1, xj2, yj2 = bboxes[j]
            area_j = max(1, (xj2 - xj1) * (yj2 - yj1))
            iomin = calcular_iomin(bboxes[i], bboxes[j])
            if iomin > iomin_thr:
                if area_i >= area_j:
                    keep[i] = False
                    break
                keep[j] = False

    out_boxes = [b for b, k in zip(bboxes, keep) if k]
    out_confs = [c for c, k in zip(confs, keep) if k]
    return out_boxes, out_confs


def nms_manual(
    bboxes: List[Tuple[int, int, int, int]],
    confs: List[float],
    iou_thr: float = 0.70,
) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    """
    NMS manual estricto:
    - ordena por confianza descendente
    - conserva la caja más confiable
    - descarta cajas con IoU > iou_thr respecto a una ya conservada
    """
    if not bboxes:
        return [], []

    order = sorted(range(len(bboxes)), key=lambda i: confs[i], reverse=True)
    keep = []

    for idx in order:
        candidate = bboxes[idx]
        should_keep = True
        for kept_idx in keep:
            iou = calcular_iou(candidate, bboxes[kept_idx])
            if iou > iou_thr:
                should_keep = False
                break
        if should_keep:
            keep.append(idx)

    out_boxes = [bboxes[i] for i in keep]
    out_confs = [confs[i] for i in keep]
    return out_boxes, out_confs


def construir_matriz_costos(
    detections: List[Dict],
    active_tracks: List[TrackState],
    frame_shape: Tuple[int, int],
    w_spatial: float = 0.8,
    w_visual: float = 0.2,
    min_cosine: float = 0.55,
    max_spatial_ratio: float = 0.25,
) -> np.ndarray:
    """
    Costo híbrido = w_spatial * distancia_espacial_normalizada + w_visual * (1 - similitud_coseno).
    """
    if not detections or not active_tracks:
        return np.zeros((len(detections), len(active_tracks)), dtype=np.float32)

    h, w = frame_shape
    diag = float(np.hypot(w, h)) + 1e-8
    max_spatial = max_spatial_ratio * diag

    C = np.full((len(detections), len(active_tracks)), 1e6, dtype=np.float32)

    static_dist_thr = 0.02 * diag

    for i, det in enumerate(detections):
        det_c = det["centroid"]
        det_e = det["embedding"]

        for j, tr in enumerate(active_tracks):
            dx = det_c[0] - tr.centroid[0]
            dy = det_c[1] - tr.centroid[1]
            d = float(np.hypot(dx, dy))
            cos = cosine_similarity_np(det_e, tr.embedding)

            if d <= static_dist_thr:
                C[i, j] = 0.01 + 0.02 * (d / (static_dist_thr + 1e-8))
                continue

            if cos < min_cosine:
                continue
            if d > max_spatial:
                continue

            spatial_cost = d / diag
            visual_cost = 1.0 - cos
            C[i, j] = w_spatial * spatial_cost + w_visual * visual_cost

    return C


def extraer_embeddings_detecciones(
    frame_bgr: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    reid_model: nn.Module,
    device: torch.device,
    transform: transforms.Compose,
) -> List[np.ndarray]:
    embeddings = []
    with torch.no_grad():
        for bbox in bboxes:
            crop_bgr = recortar_bbox(frame_bgr, bbox)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            x = transform(crop_rgb).unsqueeze(0).to(device)
            emb = reid_model(x).cpu().numpy()[0].astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings.append(emb)
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Tracking Re-ID de vacas en video con YOLOv8 + embeddings")
    parser.add_argument("--video_in", type=str, default="video_prueba.mp4")
    parser.add_argument("--video_out", type=str, default="resultado.mp4")
    parser.add_argument("--reid_model", type=str, default="models/mi_modelo_reid.pt")
    parser.add_argument("--det_conf", type=float, default=0.4)
    parser.add_argument("--sim_threshold", type=float, default=0.75)
    parser.add_argument("--ema_alpha", type=float, default=0.1)
    parser.add_argument("--max_missed", type=int, default=9999)
    parser.add_argument("--iou_threshold", type=float, default=0.70)
    args = parser.parse_args()

    device = torch.device("cpu")

    # 1) Cargar detector YOLOv8 (usamos 'm' para Medium, vital para HD y vacas camufladas)
    detector = YOLO("yolov8m.pt")
    cow_class_id = obtener_cow_class_id(detector)
    print(f"Clase 'cow' detectada en YOLO con ID: {cow_class_id}")

    # 2) Cargar extractor Re-ID.
    reid_model = cargar_reid_model(args.reid_model, device)
    reid_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 3) IO de video.
    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video de entrada: {args.video_in}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.video_out, fourcc, fps if fps > 0 else 30.0, (width, height))

    # 4) Estado del tracker híbrido.
    tracks: Dict[int, TrackState] = {}
    next_id = 1

    # Métricas operativas.
    total_frames_processados = 0
    total_detecciones_dibujadas = 0
    unique_ids = set()
    id_switches = 0
    prev_assignments: List[Tuple[Tuple[int, int, int, int], int]] = []
    track_birth_frame: Dict[int, int] = {}
    track_lifetimes: List[int] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        total_frames_processados += 1
        print(f"Frame {frame_idx}/{total_frames}")

        # 4.1) Detección YOLO por frame (imgsz=1280 para Alta Definición)
        results = detector.predict(source=frame, conf=args.det_conf, imgsz=1280, verbose=False, device="cpu")
        result = results[0]

        bboxes = []
        confs = []

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            conf = result.boxes.conf.cpu().numpy()

            for box, c, sc in zip(xyxy, cls, conf):
                if c != cow_class_id:
                    continue
                if sc <= args.det_conf:
                    continue

                x1, y1, x2, y2 = map(int, box.tolist())
                bboxes.append((x1, y1, x2, y2))
                confs.append(float(sc))

        # Filtros geométricos
        bboxes, confs = nms_manual(bboxes, confs, iou_thr=args.iou_threshold)
        bboxes, confs = filtrar_anti_mega_cajas(bboxes, confs, frame_w=width, frame_h=height)

        # 4.2) Embeddings Re-ID de detecciones.
        embeddings = extraer_embeddings_detecciones(frame, bboxes, reid_model, device, reid_transform)

        detections = []
        for bbox, emb, sc in zip(bboxes, embeddings, confs):
            detections.append(
                {
                    "bbox": bbox,
                    "centroid": bbox_centroid(bbox),
                    "embedding": emb,
                    "conf": sc,
                }
            )

        # 4.3) Seleccionar tracks activos recientes para asignación.
        active_track_ids = [tid for tid, tr in tracks.items() if (frame_idx - tr.last_seen) <= args.max_missed]
        active_tracks = [tracks[tid] for tid in active_track_ids]

        assigned_dets = set()
        assigned_tracks = set()

        if detections and active_tracks:
            C = construir_matriz_costos(
                detections=detections,
                active_tracks=active_tracks,
                frame_shape=(height, width),
                min_cosine=args.sim_threshold,
            )

            row_ind, col_ind = linear_sum_assignment(C)

            for r, c in zip(row_ind, col_ind):
                if C[r, c] >= 1e5:
                    continue

                det = detections[r]
                tr_id = active_tracks[c].track_id
                tr = tracks[tr_id]

                # EMA sobre embedding para suavizar variaciones visuales.
                new_emb = (1.0 - args.ema_alpha) * tr.embedding + args.ema_alpha * det["embedding"]
                new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-8)

                # EMA sobre coordenadas (con freno de emergencia para Cajas Viajeras)
                old_x1, old_y1, old_x2, old_y2 = tr.bbox
                det_x1, det_y1, det_x2, det_y2 = det["bbox"]
                bbox_alpha = 0.4  # Mayor reacción a la detección real
                
                old_cx, old_cy = bbox_centroid(tr.bbox)
                det_cx, det_cy = bbox_centroid(det["bbox"])
                jump_dist = float(np.hypot(det_cx - old_cx, det_cy - old_cy))

                if jump_dist > 100.0:
                    sm_x1, sm_y1, sm_x2, sm_y2 = det_x1, det_y1, det_x2, det_y2
                else:
                    sm_x1 = int((1.0 - bbox_alpha) * old_x1 + bbox_alpha * det_x1)
                    sm_y1 = int((1.0 - bbox_alpha) * old_y1 + bbox_alpha * det_y1)
                    sm_x2 = int((1.0 - bbox_alpha) * old_x2 + bbox_alpha * det_x2)
                    sm_y2 = int((1.0 - bbox_alpha) * old_y2 + bbox_alpha * det_y2)

                tr.embedding = new_emb
                tr.bbox = (sm_x1, sm_y1, sm_x2, sm_y2)
                tr.centroid = det["centroid"]
                tr.last_seen = frame_idx
                tr.missed = 0

                assigned_dets.add(r)
                assigned_tracks.add(tr_id)

        # 4.4) Detecciones no asignadas -> nuevos IDs.
        for i, det in enumerate(detections):
            if i in assigned_dets:
                continue

            tr_id = next_id
            next_id += 1
            unique_ids.add(tr_id)
            track_birth_frame[tr_id] = frame_idx
            tracks[tr_id] = TrackState(
                track_id=tr_id,
                bbox=det["bbox"],
                centroid=det["centroid"],
                embedding=det["embedding"],
                last_seen=frame_idx,
                missed=0,
            )

        # 4.5) Actualizar tracks no vistos y limpiar expirados.
        to_delete = []
        for tr_id, tr in tracks.items():
            if tr.last_seen != frame_idx:
                tr.missed += 1
            if tr.missed > args.max_missed:
                if tr_id in track_birth_frame:
                    track_lifetimes.append(frame_idx - track_birth_frame[tr_id])
                    del track_birth_frame[tr_id]
                to_delete.append(tr_id)
                continue

            # Limpieza de bordes: si sale de la pantalla, borramos el ID
            x1, y1, x2, y2 = tr.bbox
            cerca_borde = (
                x1 <= 10 or y1 <= 10 or x2 >= (width - 10) or y2 >= (height - 10)
            )
            no_vista_reciente = (frame_idx - tr.last_seen) > 30
            if cerca_borde and no_vista_reciente:
                if tr_id in track_birth_frame:
                    track_lifetimes.append(frame_idx - track_birth_frame[tr_id])
                    del track_birth_frame[tr_id]
                to_delete.append(tr_id)
                
        for tr_id in to_delete:
            del tracks[tr_id]

        # 4.6) Renderizado: Sticky Boxes garantizadas
        current_assignments: List[Tuple[Tuple[int, int, int, int], int]] = []
        for tr_id, tr in tracks.items():
            x1, y1, x2, y2 = tr.bbox
            color = color_for_id(tr_id)
            current_assignments.append((tr.bbox, tr_id))
            total_detecciones_dibujadas += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"Vaca ID: {tr_id}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        for cur_bbox, cur_id in current_assignments:
            best_iou = 0.0
            best_prev_id = None
            for prev_bbox, prev_id in prev_assignments:
                iou = calcular_iou(cur_bbox, prev_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_prev_id = prev_id
            if best_prev_id is not None and best_iou > 0.5 and best_prev_id != cur_id:
                id_switches += 1

        prev_assignments = current_assignments

        writer.write(frame)

    cap.release()
    writer.release()

    for tr_id, birth in track_birth_frame.items():
        track_lifetimes.append(max(1, frame_idx - birth + 1))

    avg_det_per_frame = (
        total_detecciones_dibujadas / total_frames_processados if total_frames_processados > 0 else 0.0
    )
    avg_track_lifetime = (
        float(np.mean(track_lifetimes)) if len(track_lifetimes) > 0 else 0.0
    )

    print("============================================================")
    print("Reporte Operativo de Tracking")
    print("============================================================")
    print(f"IDs únicos consolidados         : {len(unique_ids)}")
    print(f"Promedio detecciones por frame  : {avg_det_per_frame:.2f}")
    print(f"ID switches (aprox.)            : {id_switches}")
    print(f"Duración promedio de tracks     : {avg_track_lifetime:.2f} frames")
    print("============================================================")
    print(f"Proceso completado. Video guardado en: {args.video_out}")


if __name__ == "__main__":
    main()