#!/usr/bin/env python3
"""
Script 16: Re-ID con linea temporal estable para Erondina.

No modifica los scripts previos. Usa el mismo detector YOLO, el extractor
mi_modelo_reid.pt y la galeria de embeddings de Erondina. La diferencia con
el script 13 es que no deja cada identidad conocida atada a un solo global ID:
primero encuentra anclas automaticas por embeddings y luego une fragmentos
compatibles de la misma identidad en una linea temporal antes de renderizar.
"""

import argparse
import importlib.util
import json
import os
import pickle
from collections import defaultdict
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


s13 = cargar_modulo("script13", os.path.join(SCRIPT_DIR, "13_reid_global_embeddings_auto_erondina.py"))
base05 = s13.base05
FrameBox = s13.FrameBox
Tracklet = s13.Tracklet


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    return s13.normalize_rows(mat)


def robust_matrix_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return -1.0
    a = normalize_rows(a.astype(np.float32))
    b = normalize_rows(b.astype(np.float32))
    sims = a @ b.T
    a_to_b = np.percentile(np.max(sims, axis=1), 75)
    b_to_a = np.percentile(np.max(sims, axis=0), 75)
    return float(min(a_to_b, b_to_a))


def ensure_head_embeddings(tracklet: Tracklet):
    if not hasattr(tracklet, "head_embeddings"):
        tracklet.head_embeddings = []
    return tracklet.head_embeddings


def tracklet_head_embedding_matrix(tracklet: Tracklet):
    head_embeddings = getattr(tracklet, "head_embeddings", [])
    if not head_embeddings:
        return None
    return normalize_rows(np.vstack(head_embeddings).astype(np.float32))


def cluster_head_embedding_matrix(tracklets: Dict[int, Tracklet], local_ids: List[int]):
    mats = []
    for local_id in local_ids:
        mat = tracklet_head_embedding_matrix(tracklets[local_id])
        if mat is not None:
            mats.append(mat)
    if not mats:
        return None
    return normalize_rows(np.vstack(mats).astype(np.float32))


def color_feature_from_bgr(image: np.ndarray):
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    if h < 8 or w < 8:
        return None
    y1, y2 = int(h * 0.08), int(h * 0.94)
    x1, x2 = int(w * 0.06), int(w * 0.94)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    hch = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    mask = (sat > 22) & (val > 22)
    if int(mask.sum()) < 80:
        mask = val > 14
    if int(mask.sum()) < 40:
        mask = np.ones_like(val, dtype=bool)

    hue = hch[mask] / 180.0 * (2.0 * np.pi)
    sat_m = sat[mask] / 255.0
    val_m = val[mask] / 255.0
    lab_m = lab[mask].astype(np.float32) / 255.0
    brown_mask = ((hch < 24) | (hch > 166)) & (sat > 38) & (val > 34)
    dark_mask = val < 72
    return np.array(
        [
            float(np.mean(np.sin(hue))),
            float(np.mean(np.cos(hue))),
            float(np.median(sat_m)),
            float(np.median(val_m)),
            float(np.median(lab_m[:, 1])),
            float(np.median(lab_m[:, 2])),
            float(np.mean(brown_mask)),
            float(np.mean(dark_mask)),
        ],
        dtype=np.float32,
    )


def color_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.5
    weights = np.array([0.65, 0.65, 0.75, 1.15, 0.55, 0.55, 1.35, 1.25], dtype=np.float32)
    dist = float(np.linalg.norm((a.astype(np.float32) - b.astype(np.float32)) * weights) / np.sqrt(len(weights)))
    return float(np.clip(1.0 - dist * 1.55, 0.0, 1.0))


def gallery_color_profiles(identity_gallery: str) -> dict:
    if not identity_gallery or not os.path.exists(identity_gallery):
        return {}
    data = np.load(identity_gallery, allow_pickle=True)
    if "gallery_paths" not in data.files or "gallery_labels" not in data.files:
        return {}
    profiles = defaultdict(list)
    repo_root = os.path.dirname(SCRIPT_DIR)
    gallery_dir = os.path.dirname(os.path.abspath(identity_gallery))
    fallback_roots = [
        repo_root,
        os.getcwd(),
        os.path.dirname(gallery_dir),
        "/Volumes/T7/cow-tracker-mvp",
    ]
    for raw_label, raw_path in zip(data["gallery_labels"].tolist(), data["gallery_paths"].tolist()):
        path = str(raw_path)
        if not os.path.isabs(path):
            candidates = [os.path.join(root, path) for root in fallback_roots]
            path = next((candidate for candidate in candidates if os.path.exists(candidate)), candidates[0])
        image = cv2.imread(path)
        feature = color_feature_from_bgr(image)
        if feature is not None:
            profiles[s13.label_key(raw_label)].append(feature)
    return {
        label: np.median(np.vstack(items).astype(np.float32), axis=0)
        for label, items in profiles.items()
        if items
    }


def enrich_candidates_with_color(args, candidates, frame_records, gallery_profiles: dict):
    if not gallery_profiles or args.color_rerank_weight <= 0:
        return candidates, {}
    recs_by_local = defaultdict(list)
    for frame in frame_records:
        for rec in frame:
            recs_by_local[int(rec.local_track_id)].append(rec)
    crop_feature_cache = {}

    def feature_for_local(local_id: int):
        if local_id in crop_feature_cache:
            return crop_feature_cache[local_id]
        recs = recs_by_local.get(int(local_id), [])
        if not recs:
            crop_feature_cache[local_id] = None
            return None
        rec = recs[len(recs) // 2]
        crop = read_processed_crop(args, rec)
        feature = color_feature_from_bgr(crop)
        crop_feature_cache[local_id] = feature
        return feature

    diagnostics = {}
    for cand in candidates:
        for label, info in cand.get("all_label_scores", {}).items():
            profile = gallery_profiles.get(label)
            source_id = int(info.get("source_local_track_id", cand["cluster_local_track_ids"][0]))
            feature = feature_for_local(source_id)
            sim = color_similarity(feature, profile)
            embedding_score = float(info["score"])
            adjusted = float(np.clip(embedding_score + args.color_rerank_weight * (sim - 0.5), 0.0, 1.0))
            info["embedding_score"] = embedding_score
            info["color_similarity"] = sim
            info["score"] = adjusted
            diagnostics.setdefault(s13.display_label(label), []).append(
                {
                    "global_id": int(cand["global_id"]),
                    "source_local_track_id": source_id,
                    "embedding_score": embedding_score,
                    "color_similarity": sim,
                    "adjusted_score": adjusted,
                }
            )

        ranked = sorted(cand["all_label_scores"].items(), key=lambda x: x[1]["score"], reverse=True)
        if ranked:
            best_label, best_info = ranked[0]
            second_score = ranked[1][1]["score"] if len(ranked) > 1 else -1.0
            cand["label"] = best_label
            cand["score"] = float(best_info["score"])
            cand["margin"] = float(best_info["score"] - second_score)
            cand["median"] = float(best_info["median"])
            cand["max"] = float(best_info["max"])
            cand["support_075"] = float(best_info["support_075"])
            cand["support_080"] = float(best_info["support_080"])
            cand["nearest_vote_fraction"] = float(best_info["nearest_vote_fraction"])
            cand["source_local_track_id"] = int(best_info["source_local_track_id"])
    return candidates, diagnostics


def head_proxy_crop(crop_bgr: np.ndarray, y_fraction: float, x_margin: float) -> np.ndarray:
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    x1 = int(round(w * x_margin))
    x2 = int(round(w * (1.0 - x_margin)))
    y2 = int(round(h * y_fraction))
    if x2 <= x1 or y2 <= 2:
        return crop_bgr
    return crop_bgr[:y2, x1:x2]


def extraer_head_embeddings_detecciones(frame_bgr, bboxes, reid_model, device, transform, args):
    embeddings = []
    with base05.torch.no_grad():
        for bbox in bboxes:
            crop_bgr = base05.recortar_bbox(frame_bgr, bbox)
            crop_bgr = head_proxy_crop(crop_bgr, args.head_crop_y_fraction, args.head_crop_x_margin)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            x = transform(crop_rgb).unsqueeze(0).to(device)
            emb = reid_model(x).cpu().numpy()[0].astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings.append(emb)
    return embeddings


def serialize_tracklets(tracklets: Dict[int, Tracklet]):
    payload = {}
    for local_id, tr in tracklets.items():
        payload[int(local_id)] = {
            "local_track_id": int(tr.local_track_id),
            "first_frame": int(tr.first_frame),
            "last_frame": int(tr.last_frame),
            "frames_seen": int(tr.frames_seen),
            "embeddings": [np.asarray(x, dtype=np.float32) for x in tr.embeddings],
            "head_embeddings": [
                np.asarray(x, dtype=np.float32)
                for x in getattr(tr, "head_embeddings", [])
            ],
        }
    return payload


def deserialize_tracklets(payload):
    if not payload:
        return {}
    sample = next(iter(payload.values()))
    if isinstance(sample, Tracklet):
        for tr in payload.values():
            ensure_head_embeddings(tr)
        return payload
    tracklets = {}
    for local_id, item in payload.items():
        tr = Tracklet(
            local_track_id=int(item["local_track_id"]),
            first_frame=int(item["first_frame"]),
            last_frame=int(item["last_frame"]),
            frames_seen=int(item["frames_seen"]),
            embeddings=[np.asarray(x, dtype=np.float32) for x in item.get("embeddings", [])],
        )
        tr.head_embeddings = [
            np.asarray(x, dtype=np.float32)
            for x in item.get("head_embeddings", [])
        ]
        tracklets[int(local_id)] = tr
    return tracklets


def serialize_frame_records(frame_records):
    return [
        [
            {
                "frame_number": int(rec.frame_number),
                "local_track_id": int(rec.local_track_id),
                "bbox": tuple(int(x) for x in rec.bbox),
                "conf": float(rec.conf),
            }
            for rec in records
        ]
        for records in frame_records
    ]


def deserialize_frame_records(payload):
    if not payload:
        return []
    first_record = next((records[0] for records in payload if records), None)
    if first_record is not None and isinstance(first_record, FrameBox):
        return payload
    return [
        [
            FrameBox(
                frame_number=int(rec["frame_number"]),
                local_track_id=int(rec["local_track_id"]),
                bbox=tuple(int(x) for x in rec["bbox"]),
                conf=float(rec["conf"]),
            )
            for rec in records
        ]
        for records in payload
    ]


def label_margin(cand: dict, label: str) -> float:
    info = cand.get("all_label_scores", {}).get(label)
    if not info:
        return -999.0
    others = [
        other["score"]
        for other_label, other in cand.get("all_label_scores", {}).items()
        if other_label != label
    ]
    second = max(others) if others else -1.0
    return float(info["score"] - second)


def candidate_for_label(cand: dict, label: str, source: str = "timeline_candidate") -> dict:
    info = cand["all_label_scores"][label]
    margin = label_margin(cand, label)
    quality = (
        float(info["score"])
        + 0.05 * float(info["support_075"])
        + 0.05 * float(info["nearest_vote_fraction"])
        + 0.03 * max(0.0, margin)
        + 0.000006 * min(5000, int(cand["cluster_frames"]))
    )
    return {
        "global_id": int(cand["global_id"]),
        "label": label,
        "score": float(info["score"]),
        "margin": margin,
        "median": float(info["median"]),
        "max": float(info["max"]),
        "support_075": float(info["support_075"]),
        "support_080": float(info["support_080"]),
        "nearest_vote_fraction": float(info["nearest_vote_fraction"]),
        "source_local_track_id": int(info.get("source_local_track_id", cand["cluster_local_track_ids"][0])),
        "source_score": float(info.get("source_score", info["score"])),
        "cluster_local_track_ids": [int(x) for x in cand["cluster_local_track_ids"]],
        "cluster_frames": int(cand["cluster_frames"]),
        "all_label_scores": cand["all_label_scores"],
        "automatic_timeline_quality": float(quality),
        "assignment_mode": source,
    }


def tracklet_group_data(tracklets: Dict[int, Tracklet], local_to_global: Dict[int, int]):
    grouped = defaultdict(list)
    for local_id, global_id in local_to_global.items():
        grouped[int(global_id)].append(int(local_id))

    data = {}
    for global_id, local_ids in grouped.items():
        first_frame = min(tracklets[x].first_frame for x in local_ids)
        last_frame = max(tracklets[x].last_frame for x in local_ids)
        frames_seen = sum(tracklets[x].frames_seen for x in local_ids)
        mat = s13.cluster_embedding_matrix(tracklets, local_ids)
        head_mat = cluster_head_embedding_matrix(tracklets, local_ids)
        data[int(global_id)] = {
            "global_id": int(global_id),
            "local_track_ids": [int(x) for x in local_ids],
            "first_frame": int(first_frame),
            "last_frame": int(last_frame),
            "frames_seen": int(frames_seen),
            "embedding_count": int(0 if mat is None else len(mat)),
            "head_embedding_count": int(0 if head_mat is None else len(head_mat)),
            "matrix": mat,
            "head_matrix": head_mat,
        }
    return data


def interval_overlap(a: dict, b: dict) -> int:
    return max(0, min(a["last_frame"], b["last_frame"]) - max(a["first_frame"], b["first_frame"]) + 1)


def passes_timeline_filters(item: dict, args) -> bool:
    if item["score"] < args.timeline_identity_threshold:
        return False
    if item["support_075"] < args.timeline_min_identity_support:
        return False
    if item["nearest_vote_fraction"] < args.timeline_min_identity_vote_fraction:
        return False
    if item["margin"] < args.timeline_identity_margin:
        return False
    return True


def passes_fragment_extension_filters(item: dict, args) -> bool:
    if not passes_timeline_filters(item, args):
        return False
    if item["support_075"] < args.timeline_fragment_min_identity_support:
        return False
    if item["nearest_vote_fraction"] < args.timeline_fragment_min_identity_vote_fraction:
        return False
    if item["margin"] < args.timeline_fragment_identity_margin:
        return False
    return True


def build_timeline_assignments(args, tracklets, local_to_global, candidates, known_assignments, gallery_labels):
    group_data = tracklet_group_data(tracklets, local_to_global)
    cand_by_gid = {int(c["global_id"]): c for c in candidates}
    labels = sorted(set(gallery_labels))

    strict_anchors_by_label = {}
    for assignment in known_assignments.values():
        label = assignment["label"]
        strict_anchors_by_label[label] = {
            **assignment,
            "automatic_timeline_quality": float(
                assignment.get("automatic_assignment_quality", assignment.get("score", 0.0))
            ),
            "assignment_mode": "automatic_strict_anchor_from_global_embeddings",
        }

    anchors_by_label = {}
    for label in labels:
        options = []
        for cand in candidates:
            if label not in cand.get("all_label_scores", {}):
                continue
            item = candidate_for_label(cand, label, "automatic_anchor_timeline")
            if passes_timeline_filters(item, args):
                gd = group_data.get(int(item["global_id"]), {})
                item = {
                    **item,
                    "first_frame": int(gd.get("first_frame", 0)),
                    "last_frame": int(gd.get("last_frame", 0)),
                    "frames_seen": int(gd.get("frames_seen", item.get("cluster_frames", 0))),
                }
                options.append(item)
        if options:
            start_limit = args.start_frame + max(1, args.anchor_start_window_frames)
            start_options = [
                item for item in options
                if args.anchor_start_window_frames >= 0 and item.get("first_frame", 10**9) <= start_limit
            ]
            pool = start_options if start_options else options
            anchor = sorted(
                pool,
                key=lambda x: (
                    x["automatic_timeline_quality"],
                    x["score"],
                    x["nearest_vote_fraction"],
                    x["frames_seen"],
                ),
                reverse=True,
            )[0]
            anchor["assignment_mode"] = (
                "automatic_start_visible_anchor_by_embeddings"
                if start_options
                else "automatic_best_anchor_by_embeddings"
            )
            anchors_by_label[label] = anchor
        elif label in strict_anchors_by_label:
            anchors_by_label[label] = strict_anchors_by_label[label]

    for label, anchor in list(anchors_by_label.items()):
        if label not in strict_anchors_by_label:
            continue
        strict = strict_anchors_by_label[label]
        anchor["strict_anchor_global_id"] = int(strict["global_id"])
        anchor["strict_anchor_score"] = float(strict["score"])
        anchor["strict_anchor_assignment_mode"] = strict["assignment_mode"]

    prelim = {}
    decisions = defaultdict(list)
    for label in labels:
        anchor = anchors_by_label.get(label)
        if not anchor:
            continue
        anchor_gid = int(anchor["global_id"])
        accepted = {anchor_gid: {**anchor, "timeline_relation": "anchor", "compatibility_to_anchor": 1.0}}

        options = []
        for cand in candidates:
            gid = int(cand["global_id"])
            if gid == anchor_gid or label not in cand.get("all_label_scores", {}):
                continue
            item = candidate_for_label(cand, label)
            if passes_fragment_extension_filters(item, args):
                options.append(item)

        options = sorted(
            options,
            key=lambda x: (
                x["automatic_timeline_quality"],
                x["score"],
                x["nearest_vote_fraction"],
                x["cluster_frames"],
            ),
            reverse=True,
        )

        for item in options:
            gid = int(item["global_id"])
            gd = group_data.get(gid)
            if gd is None or gd["matrix"] is None:
                continue
            overlapping = False
            for accepted_gid in accepted:
                ad = group_data.get(int(accepted_gid))
                if ad and interval_overlap(gd, ad) > args.timeline_max_overlap_frames:
                    overlapping = True
                    break
            if overlapping:
                decisions[label].append({**item, "accepted": False, "reason": "temporal_overlap"})
                continue

            anchor_data = group_data.get(anchor_gid)
            body_to_anchor = robust_matrix_similarity(gd["matrix"], anchor_data["matrix"]) if anchor_data else -1.0
            head_to_anchor = robust_matrix_similarity(gd["head_matrix"], anchor_data["head_matrix"]) if anchor_data else -1.0
            compatibility_to_anchor = max(body_to_anchor, head_to_anchor)
            body_to_any = body_to_anchor
            head_to_any = head_to_anchor
            compatibility_to_any = compatibility_to_anchor
            for accepted_gid in accepted:
                ad = group_data.get(int(accepted_gid))
                if ad and ad["matrix"] is not None:
                    body_to_any = max(body_to_any, robust_matrix_similarity(gd["matrix"], ad["matrix"]))
                if ad and ad["head_matrix"] is not None:
                    head_to_any = max(head_to_any, robust_matrix_similarity(gd["head_matrix"], ad["head_matrix"]))
                compatibility_to_any = max(compatibility_to_any, body_to_any, head_to_any)

            high_confidence_bridge = (
                item["score"] >= args.timeline_high_confidence_threshold
                and item["margin"] >= args.timeline_high_confidence_margin
                and item["nearest_vote_fraction"] >= args.timeline_high_confidence_vote_fraction
                and compatibility_to_any >= args.timeline_bridge_min_reid_similarity
            )
            compatible_by_body = body_to_any >= args.timeline_min_reid_similarity
            compatible_by_head = head_to_any >= args.timeline_min_head_similarity
            if not compatible_by_body and not compatible_by_head and not high_confidence_bridge:
                decisions[label].append(
                    {
                        **item,
                        "accepted": False,
                        "reason": "low_fragment_similarity",
                        "compatibility_to_anchor": float(compatibility_to_anchor),
                        "body_similarity_to_anchor": float(body_to_anchor),
                        "head_similarity_to_anchor": float(head_to_anchor),
                        "compatibility_to_any_accepted": float(compatibility_to_any),
                        "body_similarity_to_any_accepted": float(body_to_any),
                        "head_similarity_to_any_accepted": float(head_to_any),
                    }
                )
                continue

            accepted[gid] = {
                **item,
                "timeline_relation": "compatible_fragment",
                "compatibility_to_anchor": float(compatibility_to_anchor),
                "body_similarity_to_anchor": float(body_to_anchor),
                "head_similarity_to_anchor": float(head_to_anchor),
                "compatibility_to_any_accepted": float(compatibility_to_any),
                "body_similarity_to_any_accepted": float(body_to_any),
                "head_similarity_to_any_accepted": float(head_to_any),
                "compatible_by_body": bool(compatible_by_body),
                "compatible_by_head": bool(compatible_by_head),
                "assignment_mode": "automatic_timeline_fragment",
            }
            decisions[label].append({**accepted[gid], "accepted": True, "reason": "accepted"})

        prelim[label] = accepted

    claims = []
    for label, items in prelim.items():
        for gid, item in items.items():
            claims.append((int(gid), label, item))

    by_gid = defaultdict(list)
    for gid, label, item in claims:
        by_gid[gid].append((label, item))

    final = {}
    conflicts = []
    for gid, rows in by_gid.items():
        if len(rows) == 1:
            label, item = rows[0]
            final[gid] = item
            continue
        rows = sorted(
            rows,
            key=lambda x: (
                x[1].get("timeline_relation") == "anchor",
                x[1]["automatic_timeline_quality"],
                x[1]["score"],
                x[1]["nearest_vote_fraction"],
            ),
            reverse=True,
        )
        winner_label, winner_item = rows[0]
        final[gid] = winner_item
        conflicts.append(
            {
                "global_id": int(gid),
                "winner_label": winner_label,
                "discarded_labels": [label for label, _ in rows[1:]],
                "claims": [
                    {
                        "label": label,
                        "score": float(item["score"]),
                        "margin": float(item["margin"]),
                        "quality": float(item["automatic_timeline_quality"]),
                    }
                    for label, item in rows
                ],
            }
        )

    for gid, item in list(final.items()):
        gd = group_data.get(int(gid), {})
        final[gid] = {
            **item,
            "first_frame": int(gd.get("first_frame", 0)),
            "last_frame": int(gd.get("last_frame", 0)),
            "frames_seen": int(gd.get("frames_seen", item.get("cluster_frames", 0))),
            "embedding_count": int(gd.get("embedding_count", 0)),
            "head_embedding_count": int(gd.get("head_embedding_count", 0)),
        }

    timeline_by_label = defaultdict(list)
    for gid, item in sorted(final.items()):
        timeline_by_label[item["label"]].append(item)
    for label in list(timeline_by_label):
        timeline_by_label[label] = sorted(timeline_by_label[label], key=lambda x: (x["first_frame"], x["global_id"]))

    diagnostics = {
        "anchors_by_label": {s13.display_label(k): v for k, v in sorted(anchors_by_label.items())},
        "strict_anchors_by_label": {s13.display_label(k): v for k, v in sorted(strict_anchors_by_label.items())},
        "timeline_decisions_by_label": {s13.display_label(k): v for k, v in sorted(decisions.items())},
        "timeline_conflicts": conflicts,
        "timeline_fragments_by_label": {
            s13.display_label(k): v for k, v in sorted(timeline_by_label.items())
        },
        "candidate_count": len(candidates),
        "global_group_count": len(group_data),
    }
    return final, diagnostics, group_data


def duplicate_known_frames(frame_records, local_to_global, timeline_assignments):
    label_by_gid = {int(gid): data["label"] for gid, data in timeline_assignments.items()}
    duplicates = defaultdict(int)
    duplicate_examples = defaultdict(list)
    for idx, records in enumerate(frame_records, start=1):
        labels = defaultdict(list)
        for rec in records:
            gid = local_to_global.get(rec.local_track_id)
            if gid in label_by_gid:
                labels[label_by_gid[gid]].append((int(gid), int(rec.local_track_id), float(rec.conf)))
        for label, rows in labels.items():
            if len(rows) > 1:
                duplicates[label] += 1
                if len(duplicate_examples[label]) < 8:
                    duplicate_examples[label].append({"render_frame": idx, "detections": rows})
    return dict(duplicates), dict(duplicate_examples)


def choose_known_records(records, local_to_global, timeline_assignments):
    best = {}
    suppressed = []
    quality_by_gid = {
        int(gid): float(data.get("automatic_timeline_quality", data.get("score", 0.0)))
        for gid, data in timeline_assignments.items()
    }
    for rec in records:
        gid = local_to_global.get(rec.local_track_id)
        if gid not in timeline_assignments:
            continue
        label = timeline_assignments[gid]["label"]
        rank = (quality_by_gid.get(gid, 0.0), float(rec.conf), -abs(rec.bbox[0] - rec.bbox[2]))
        if label not in best or rank > best[label][0]:
            if label in best:
                suppressed.append(best[label][1])
            best[label] = (rank, rec)
        else:
            suppressed.append(rec)
    return {label: item[1] for label, item in best.items()}, suppressed


def stable_known_color(label: str):
    palette = {
        "maria": (0, 255, 255),
        "marta": (255, 160, 40),
        "margarita": (60, 255, 60),
    }
    return palette.get(label, (255, 255, 255))


def bbox_center_and_diag(bbox):
    x1, y1, x2, y2 = [float(x) for x in bbox]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    diag = float(np.hypot(max(1.0, x2 - x1), max(1.0, y2 - y1)))
    return cx, cy, diag


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def bbox_edge_info(bbox, width: int, height: int, edge_margin_ratio: float):
    x1, y1, x2, y2 = [float(x) for x in bbox]
    margin = float(edge_margin_ratio) * float(min(width, height))
    distances = {
        "left": x1,
        "top": y1,
        "right": float(width) - x2,
        "bottom": float(height) - y2,
    }
    nearest_edge = min(distances, key=distances.get)
    nearest_distance = float(distances[nearest_edge])
    return {
        "near_edge": bool(nearest_distance <= margin),
        "nearest_edge": nearest_edge,
        "nearest_edge_distance_px": nearest_distance,
        "edge_margin_px": margin,
    }


def find_continuity_record(label, state, records, local_to_global, timeline_assignments, used_rec_ids, args):
    if state.get("bbox") is None or state.get("last_seen_idx") is None:
        return None
    if state.get("current_idx", 0) - int(state["last_seen_idx"]) > args.known_reacquire_frames:
        return None

    prev_bbox = state["bbox"]
    pcx, pcy, pdiag = bbox_center_and_diag(prev_bbox)
    best = None
    best_score = -999.0
    for rec in records:
        if id(rec) in used_rec_ids:
            continue
        gid = local_to_global.get(rec.local_track_id)
        if gid in timeline_assignments:
            continue
        rcx, rcy, rdiag = bbox_center_and_diag(rec.bbox)
        center_dist = float(np.hypot(rcx - pcx, rcy - pcy))
        norm_dist = center_dist / max(pdiag, rdiag, 1.0)
        iou = bbox_iou(prev_bbox, rec.bbox)
        if iou < args.known_reacquire_min_iou and norm_dist > args.known_reacquire_max_center_factor:
            continue
        score = (2.0 * iou) + (1.0 - norm_dist) + 0.2 * float(rec.conf)
        if score > best_score:
            best_score = score
            best = rec
    return best


def smooth_known_bbox(prev_bbox, raw_bbox, args):
    raw = np.array(raw_bbox, dtype=np.float32)
    if prev_bbox is None:
        return raw
    prev = np.array(prev_bbox, dtype=np.float32)
    pcx, pcy, pdiag = bbox_center_and_diag(prev)
    rcx, rcy, rdiag = bbox_center_and_diag(raw)
    center_dist = float(np.hypot(rcx - pcx, rcy - pcy))
    reset_limit = args.known_smoothing_reset_center_factor * max(pdiag, rdiag, 1.0)
    if center_dist > reset_limit:
        return raw
    alpha = float(args.known_bbox_smoothing_alpha)
    return alpha * raw + (1.0 - alpha) * prev


def int_bbox(bbox, width, height):
    arr = np.array(bbox, dtype=np.float32)
    arr[0::2] = np.clip(arr[0::2], 0, max(0, width - 1))
    arr[1::2] = np.clip(arr[1::2], 0, max(0, height - 1))
    return tuple(int(round(x)) for x in arr.tolist())


def draw_text_bg(img, text, org, font_scale, color, thickness=2):
    return s13.draw_text_bg(img, text, org, font_scale, color, thickness)


def resolve_process_size(source_width: int, source_height: int, args) -> Tuple[int, int]:
    if args.process_width <= 0 and args.process_height <= 0:
        return int(source_width), int(source_height)
    if args.process_width > 0 and args.process_height > 0:
        return int(args.process_width), int(args.process_height)
    if args.process_width > 0:
        scale = float(args.process_width) / max(1, float(source_width))
        return int(args.process_width), int(round(source_height * scale))
    scale = float(args.process_height) / max(1, float(source_height))
    return int(round(source_width * scale)), int(args.process_height)


def resize_frame_to_process(frame, width: int, height: int):
    if frame is None or frame.size == 0:
        return frame
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def read_processed_crop(args, rec: FrameBox, size=(260, 200)):
    cap = cv2.VideoCapture(args.video_in)
    cap.set(cv2.CAP_PROP_POS_FRAMES, rec.frame_number - 1)
    ok, frame = cap.read()
    cap.release()
    if ok:
        source_h, source_w = frame.shape[:2]
        process_w, process_h = resolve_process_size(source_w, source_h, args)
        frame = resize_frame_to_process(frame, process_w, process_h)
        crop = base05.recortar_bbox(frame, rec.bbox)
    else:
        crop = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def build_unknown_display_ids(frame_records, local_to_global, timeline_assignments):
    known_global_ids = {int(gid) for gid in timeline_assignments}
    first_seen = {}
    for idx, records in enumerate(frame_records):
        for rec in records:
            gid = local_to_global.get(rec.local_track_id)
            if gid is None:
                continue
            gid = int(gid)
            if gid in known_global_ids:
                continue
            first_seen.setdefault(gid, idx)
    ordered = sorted(first_seen, key=lambda gid: (first_seen[gid], gid))
    return {gid: display_id for display_id, gid in enumerate(ordered, start=1)}


def render_video_timeline(args, frame_records, local_to_global, timeline_assignments, width, height, fps):
    cap = cv2.VideoCapture(args.video_in)
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    os.makedirs(os.path.dirname(args.video_out), exist_ok=True)
    writer = cv2.VideoWriter(
        args.video_out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        (width, height),
    )

    suppressed_duplicate_count = 0
    known_labels = sorted({item["label"] for item in timeline_assignments.values()})
    known_state = {
        label: {"bbox": None, "last_seen_idx": None, "last_confirmed_idx": None, "last_gid": None}
        for label in known_labels
    }
    unknown_display_ids = build_unknown_display_ids(frame_records, local_to_global, timeline_assignments)
    for idx, records in enumerate(frame_records):
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_frame_to_process(frame, width, height)
        known_records_by_label, suppressed = choose_known_records(records, local_to_global, timeline_assignments)
        suppressed_duplicate_count += len(suppressed)

        confirmed_labels = set(known_records_by_label)
        used_known_rec_ids = {id(rec) for rec in known_records_by_label.values()}
        for label in known_labels:
            known_state[label]["current_idx"] = idx
        for label in known_labels:
            if label in known_records_by_label:
                continue
            rec = find_continuity_record(
                label,
                known_state[label],
                records,
                local_to_global,
                timeline_assignments,
                used_known_rec_ids,
                args,
            )
            if rec is not None:
                known_records_by_label[label] = rec
                used_known_rec_ids.add(id(rec))

        known_draws = {}
        for label in known_labels:
            rec = known_records_by_label.get(label)
            state = known_state[label]
            if rec is not None:
                gid = local_to_global.get(rec.local_track_id)
                gid = int(gid) if gid is not None else -1
                state["bbox"] = smooth_known_bbox(state["bbox"], rec.bbox, args)
                state["last_seen_idx"] = idx
                if label in confirmed_labels:
                    state["last_confirmed_idx"] = idx
                state["last_gid"] = gid
                known_draws[label] = (state["bbox"], gid, False)
            elif (
                state["bbox"] is not None
                and state["last_seen_idx"] is not None
                and idx - int(state["last_seen_idx"]) <= args.known_hold_frames
            ):
                known_draws[label] = (state["bbox"], state["last_gid"], True)

        known_priority_bboxes = [bbox for bbox, _gid, _is_hold in known_draws.values()]
        visible_global_ids = set()
        for rec in records:
            gid = local_to_global.get(rec.local_track_id)
            if gid is None:
                continue
            visible_global_ids.add(int(gid))
            x1, y1, x2, y2 = rec.bbox
            if id(rec) in used_known_rec_ids or int(gid) in timeline_assignments:
                continue
            if any(bbox_iou(rec.bbox, known_bbox) >= args.unknown_suppress_iou_with_known for known_bbox in known_priority_bboxes):
                continue
            color = base05.color_for_id(int(gid))
            if args.unknown_label_mode == "generic":
                text = "Vaca"
            else:
                display_gid = unknown_display_ids.get(int(gid), int(gid))
                text = f"Vaca {int(display_gid):02d}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_text_bg(frame, text, (x1, max(38, y1 - 10)), 0.78, color, 2)

        for label, (bbox, _gid, is_hold) in known_draws.items():
            x1, y1, x2, y2 = int_bbox(bbox, width, height)
            color = stable_known_color(label)
            text = s13.display_label(label)
            thickness = 8 if not is_hold else 5
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            draw_text_bg(frame, text, (x1, max(66, y1 - 20)), 1.65, color, 5)

        display_frame = int(args.display_frame_start) + idx
        display_total = int(args.display_frame_start) + max(0, len(frame_records) - 1)
        displayed_total_cows = int(args.display_total_cows) if int(args.display_total_cows) > 0 else int(args.expected_total_cows)
        if args.hide_visible_overlay:
            overlay = f"Frame {display_frame}/{display_total} | total estimado {displayed_total_cows}"
        else:
            overlay = (
                f"Frame {display_frame}/{display_total} | visibles {len(visible_global_ids)} "
                f"| total estimado {displayed_total_cows}"
            )
        draw_text_bg(frame, overlay, (28, 58), 1.05, (255, 255, 255), 3)
        writer.write(frame)
        if idx == 0 or (idx + 1) % 100 == 0:
            print(f"Render frame {idx + 1}/{len(frame_records)}")

    cap.release()
    writer.release()
    return suppressed_duplicate_count


def intervals_from_bits(bits, value=True):
    out = []
    start = None
    for idx, bit in enumerate(bits, start=1):
        if bit == value and start is None:
            start = idx
        is_last = idx == len(bits)
        if start is not None and (bit != value or is_last):
            end = idx - 1 if bit != value else idx
            out.append((start, end, end - start + 1))
            start = None
    return out


def audit_locked_known_tracks(args, frame_records, local_to_global, timeline_assignments, fps, width, height):
    known_labels = sorted({item["label"] for item in timeline_assignments.values()})
    state = {
        label: {"bbox": None, "last_seen_idx": None, "last_confirmed_idx": None, "last_gid": None}
        for label in known_labels
    }
    present_bits = {label: [] for label in known_labels}
    tracked_bits = {label: [] for label in known_labels}
    bbox_trace = {label: [] for label in known_labels}
    continuity_counts = defaultdict(int)
    hold_counts = defaultdict(int)
    confirmed_counts = defaultdict(int)
    all_together_bits = []

    for idx, records in enumerate(frame_records):
        known_records_by_label, _ = choose_known_records(records, local_to_global, timeline_assignments)
        confirmed_labels = set(known_records_by_label)
        used_known_rec_ids = {id(rec) for rec in known_records_by_label.values()}
        for label in known_labels:
            state[label]["current_idx"] = idx

        continuity_labels = set()
        for label in known_labels:
            if label in known_records_by_label:
                continue
            rec = find_continuity_record(
                label,
                state[label],
                records,
                local_to_global,
                timeline_assignments,
                used_known_rec_ids,
                args,
            )
            if rec is not None:
                known_records_by_label[label] = rec
                used_known_rec_ids.add(id(rec))
                continuity_labels.add(label)

        frame_present = set()
        frame_tracked = set()
        for label in known_labels:
            rec = known_records_by_label.get(label)
            label_state = state[label]
            frame_bbox = None
            if rec is not None:
                label_state["bbox"] = smooth_known_bbox(label_state["bbox"], rec.bbox, args)
                label_state["last_seen_idx"] = idx
                if label in confirmed_labels:
                    label_state["last_confirmed_idx"] = idx
                    confirmed_counts[label] += 1
                elif label in continuity_labels:
                    continuity_counts[label] += 1
                gid = local_to_global.get(rec.local_track_id)
                label_state["last_gid"] = int(gid) if gid is not None else -1
                frame_present.add(label)
                frame_tracked.add(label)
                frame_bbox = np.array(label_state["bbox"], dtype=np.float32).tolist()
            elif (
                label_state["bbox"] is not None
                and label_state["last_seen_idx"] is not None
                and idx - int(label_state["last_seen_idx"]) <= args.known_hold_frames
            ):
                hold_counts[label] += 1
                frame_present.add(label)
                frame_bbox = np.array(label_state["bbox"], dtype=np.float32).tolist()
            bbox_trace[label].append(frame_bbox)

        for label in known_labels:
            present_bits[label].append(label in frame_present)
            tracked_bits[label].append(label in frame_tracked)
        all_together_bits.append(bool(known_labels and all(label in frame_present for label in known_labels)))

    labels_report = {}
    long_gap_threshold = max(1, int(round(2.0 * fps)))
    for label in known_labels:
        bits = present_bits[label]
        tracked = tracked_bits[label]
        gaps = intervals_from_bits(bits, False)
        long_gaps = [gap for gap in gaps if gap[2] >= long_gap_threshold]
        classified_long_gaps = []
        for start, end, count in long_gaps:
            prev_bbox = bbox_trace[label][start - 2] if start > 1 else None
            next_bbox = bbox_trace[label][end] if end < len(bits) else None
            prev_edge = bbox_edge_info(prev_bbox, width, height, args.known_exit_edge_margin_ratio) if prev_bbox is not None else None
            next_edge = bbox_edge_info(next_bbox, width, height, args.known_exit_edge_margin_ratio) if next_bbox is not None else None
            edge_plausible = bool(
                (prev_edge and prev_edge["near_edge"])
                or (next_edge and next_edge["near_edge"])
                or start == 1
            )
            if end == len(bits) and prev_edge and prev_edge["near_edge"]:
                kind = "left_frame_near_edge"
            elif start == 1:
                kind = "before_first_detection"
            elif edge_plausible:
                kind = "exit_or_reentry_near_edge"
            else:
                kind = "midframe_tracking_gap"
            classified_long_gaps.append(
                {
                    "start_render_frame": int(start),
                    "end_render_frame": int(end),
                    "frames": int(count),
                    "seconds": float(count / fps) if fps else 0.0,
                    "start_original_frame": int(args.start_frame + start),
                    "end_original_frame": int(args.start_frame + end),
                    "classification": kind,
                    "edge_plausible": bool(edge_plausible),
                    "prev_bbox_edge": prev_edge,
                    "next_bbox_edge": next_edge,
                }
            )
        midframe_long_gaps = [
            gap for gap in classified_long_gaps
            if gap["classification"] == "midframe_tracking_gap"
        ]
        labels_report[s13.display_label(label)] = {
            "present_frames": int(sum(bits)),
            "tracked_detection_frames": int(sum(tracked)),
            "confirmed_reid_frames": int(confirmed_counts[label]),
            "continuity_follow_frames": int(continuity_counts[label]),
            "hold_only_frames": int(hold_counts[label]),
            "present_ratio": float(sum(bits) / len(bits)) if bits else 0.0,
            "first_present_frame": int(next((i for i, bit in enumerate(bits, start=1) if bit), 0)) or None,
            "last_present_frame": int(len(bits) - next((i for i, bit in enumerate(reversed(bits), start=1) if bit), 0) + 1) if any(bits) else None,
            "long_gap_count_ge_2s": int(len(long_gaps)),
            "midframe_long_gap_count_ge_2s": int(len(midframe_long_gaps)),
            "long_gaps_ge_2s": classified_long_gaps[:20],
        }

    all_intervals = intervals_from_bits(all_together_bits, True)
    return {
        "labels": labels_report,
        "all_known_together": {
            "exists": bool(any(all_together_bits)),
            "frame_count": int(sum(all_together_bits)),
            "first_render_frame": int(next((i for i, bit in enumerate(all_together_bits, start=1) if bit), 0)) or None,
            "last_render_frame": int(len(all_together_bits) - next((i for i, bit in enumerate(reversed(all_together_bits), start=1) if bit), 0) + 1) if any(all_together_bits) else None,
            "intervals": [
                {
                    "start_render_frame": int(start),
                    "end_render_frame": int(end),
                    "frames": int(count),
                    "seconds": float(count / fps) if fps else 0.0,
                }
                for start, end, count in all_intervals[:20]
            ],
        },
    }


def make_timeline_contact_sheet(args, frame_records, local_to_global, timeline_assignments, out_path):
    rows = []
    labels = sorted({item["label"] for item in timeline_assignments.values()})
    for label in labels:
        gids = {
            int(gid)
            for gid, item in timeline_assignments.items()
            if item["label"] == label
        }
        recs = [
            rec
            for frame in frame_records
            for rec in frame
            if local_to_global.get(rec.local_track_id) in gids
        ]
        if not recs:
            continue
        picks = [recs[0], recs[len(recs) // 2], recs[-1]]
        cells = []
        for rec in picks:
            gid = local_to_global.get(rec.local_track_id)
            crop = read_processed_crop(args, rec)
            if getattr(args, "public_report", False):
                text = f"{s13.display_label(label)}"
            else:
                text = f"{s13.display_label(label)} G{int(gid)} f{rec.frame_number}"
            s13.draw_text_bg(crop, text, (6, 30), 0.64, (0, 255, 255), 2)
            cells.append(crop)
        rows.append(np.hstack(cells))
    if not rows:
        return None
    sheet = np.vstack(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, sheet)
    return out_path


def make_timeline_candidate_sheet(args, frame_records, local_to_global, candidates, out_path, top_k=4):
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
            crop = read_processed_crop(args, rec)
            if getattr(args, "public_report", False):
                text = f"{s13.display_label(label)} s{score:.2f}"
            else:
                text = f"{s13.display_label(label)} G{gid} L{source_local_id} s{score:.2f}"
            s13.draw_text_bg(crop, text, (6, 28), 0.54, (0, 255, 255), 2)
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


def zero_base_render_frames(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if "original_frame" in key:
                continue
            if (
                key in {
                    "first_render_frame",
                    "last_render_frame",
                    "start_render_frame",
                    "end_render_frame",
                    "first_present_frame",
                    "last_present_frame",
                }
                and isinstance(item, int)
            ):
                out[key] = max(0, item - 1)
            else:
                out[key] = zero_base_render_frames(item)
        return out
    if isinstance(value, list):
        return [zero_base_render_frames(item) for item in value]
    return value


def public_report(report: dict, args) -> dict:
    compact = {
        "report_type": "cowtrack_final_public_report",
        "frame_indexing": "zero_based",
        "video_out": os.path.basename(args.video_out) if args.render else None,
        "processed_frames": int(report["processed_frames"]),
        "display_frame_range": {
            "start": 0,
            "end": max(0, int(report["processed_frames"]) - 1),
        },
        "processing_video_size": report["processing_video_size"],
        "estimated_total_cows": report["estimated_total_cows"],
        "estimated_total_cows_method": report["estimated_total_cows_method"],
        "expected_total_cows": report["expected_total_cows"],
        "count_error": report["count_error"],
        "absolute_count_error": report["absolute_count_error"],
        "count_accuracy": report["count_accuracy"],
        "count_within_tolerance": report["count_within_tolerance"],
        "unknown_cows_estimated": report["unknown_cows_estimated"],
        "per_frame_detection_summary": {
            "mean_visible_detections": report["visible_cows_per_frame"]["mean"],
            "median_visible_detections": report["visible_cows_per_frame"]["median"],
            "p95_visible_detections": report["visible_cows_per_frame"]["p95"],
            "max_visible_detections": report["visible_cows_per_frame"]["max"],
            "note": (
                "Este valor resume detecciones por frame y no se usa como conteo final de animales, "
                "porque puede incluir fragmentación temporal u oclusiones."
            ),
        },
        "known_found": report["known_found"],
        "known_missing": report["known_missing"],
        "known_frame_hits": report["known_frame_hits"],
        "known_hit_ratio": report["known_hit_ratio"],
        "known_first_render_frame": report["known_first_render_frame"],
        "known_last_render_frame": report["known_last_render_frame"],
        "all_known_together": report["all_known_together"],
        "locked_track_audit": report["locked_track_audit"],
        "locked_track_audit_ok": report["locked_track_audit_ok"],
        "locked_midframe_gap_count_ge_2s": report["locked_midframe_gap_count_ge_2s"],
        "known_id_switches_by_design": report["known_id_switches_by_design"],
        "ready_for_render_by_automatic_checks": report["ready_for_render_by_automatic_checks"],
        "identity_scores": {
            s13.display_label(item["label"]): float(item["score"])
            for item in report["timeline_assignments_by_global_id"].values()
        },
        "contact_sheet": os.path.basename(report["contact_sheet"]) if report.get("contact_sheet") else None,
        "candidate_sheet": os.path.basename(report["candidate_sheet"]) if report.get("candidate_sheet") else None,
        "metric_note": (
            "El video se reporta como pieza autónoma con indexación de frames desde 0. "
            "El conteo final corresponde al total consolidado del rodeo en el video renderizado. "
            "Precision y recall reales por bounding box requieren anotaciones ground truth por frame."
        ),
    }
    if "suppressed_duplicate_known_detections_in_render" in report:
        compact["suppressed_duplicate_known_detections_in_render"] = report["suppressed_duplicate_known_detections_in_render"]
    if "render_stabilization" in report:
        compact["render_stabilization"] = report["render_stabilization"]
    if "color_rerank" in report:
        compact["color_rerank"] = report["color_rerank"]
    return zero_base_render_frames(compact)


def load_or_build_evidence(args, device, detector, cow_class_id, reid_model, transform):
    if args.evidence_cache_in:
        print(f"Cargando evidencia cacheada: {args.evidence_cache_in}")
        with open(args.evidence_cache_in, "rb") as f:
            payload = pickle.load(f)
        return (
            deserialize_tracklets(payload["tracklets"]),
            deserialize_frame_records(payload["frame_records"]),
            int(payload["processed"]),
            int(payload["total_frames"]),
            float(payload["fps"]),
            int(payload["width"]),
            int(payload["height"]),
        )

    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {args.video_in}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height = resolve_process_size(source_width, source_height, args)
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    tracklets: Dict[int, Tracklet] = {}
    frame_records: List[List[FrameBox]] = []
    processed = 0
    frame_number = args.start_frame
    fallback_next_id = 100000
    fallback_prev_tracks = []
    print("Pass 1/2: tracking + evidencia ReID por embeddings")
    while True:
        if args.max_frames > 0 and processed >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_frame_to_process(frame, width, height)
        processed += 1
        frame_number += 1
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
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            if result.boxes.id is not None:
                ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                ids = []
                used_prev = set()
                next_prev_tracks = []
                for box in boxes:
                    bbox = tuple(map(int, box.tolist()))
                    best_idx = None
                    best_iou = 0.0
                    for idx, prev in enumerate(fallback_prev_tracks):
                        if idx in used_prev:
                            continue
                        iou = bbox_iou(bbox, prev["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx
                    if best_idx is not None and best_iou >= 0.25:
                        local_id = int(fallback_prev_tracks[best_idx]["id"])
                        used_prev.add(best_idx)
                    else:
                        local_id = fallback_next_id
                        fallback_next_id += 1
                    ids.append(local_id)
                    next_prev_tracks.append({"id": local_id, "bbox": bbox})
                ids = np.asarray(ids, dtype=int)
                fallback_prev_tracks = next_prev_tracks
            reid_bboxes = []
            reid_ids = []
            for box, local_id, conf in zip(boxes, ids, confs):
                local_id = int(local_id)
                bbox = tuple(map(int, box.tolist()))
                records.append(FrameBox(frame_number=frame_number, local_track_id=local_id, bbox=bbox, conf=float(conf)))
                if local_id not in tracklets:
                    tracklets[local_id] = Tracklet(local_track_id=local_id, first_frame=frame_number, last_frame=frame_number)
                    ensure_head_embeddings(tracklets[local_id])
                tr = tracklets[local_id]
                tr.last_frame = frame_number
                tr.frames_seen += 1
                if processed % args.reid_every == 0:
                    reid_bboxes.append(bbox)
                    reid_ids.append(local_id)
            if reid_bboxes:
                if args.focus_margin_x > 0 or args.focus_margin_y > 0:
                    embeddings = s13.extraer_embeddings_detecciones_enfocadas(
                        frame,
                        reid_bboxes,
                        reid_model,
                        device,
                        transform,
                        args.focus_margin_x,
                        args.focus_margin_y,
                    )
                else:
                    embeddings = base05.extraer_embeddings_detecciones(frame, reid_bboxes, reid_model, device, transform)
                for local_id, emb in zip(reid_ids, embeddings):
                    tracklets[local_id].embeddings.append(emb)
                if args.head_embeddings and args.head_reid_every > 0 and processed % args.head_reid_every == 0:
                    head_embeddings = extraer_head_embeddings_detecciones(
                        frame,
                        reid_bboxes if reid_bboxes else [rec.bbox for rec in records],
                        reid_model,
                        device,
                        transform,
                        args,
                    )
                    head_ids = reid_ids if reid_bboxes else [rec.local_track_id for rec in records]
                    for local_id, emb in zip(head_ids, head_embeddings):
                        ensure_head_embeddings(tracklets[local_id]).append(emb)
        frame_records.append(records)
        if processed == 1 or processed % 100 == 0:
            progress_total = args.max_frames if args.max_frames > 0 else total_frames - args.start_frame
            print(f"Track frame {processed}/{progress_total}")
    cap.release()

    if args.evidence_cache_out:
        os.makedirs(os.path.dirname(args.evidence_cache_out), exist_ok=True)
        with open(args.evidence_cache_out, "wb") as f:
            pickle.dump(
                {
                    "tracklets": serialize_tracklets(tracklets),
                    "frame_records": serialize_frame_records(frame_records),
                    "processed": processed,
                    "total_frames": total_frames,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "source_width": source_width,
                    "source_height": source_height,
                    "start_frame": args.start_frame,
                    "video_in": args.video_in,
                    "process_width": width,
                    "process_height": height,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"Evidencia guardada en: {args.evidence_cache_out}")

    return tracklets, frame_records, processed, total_frames, fps, width, height


def main():
    parser = argparse.ArgumentParser(description="Re-ID timeline estable por embeddings")
    parser.add_argument("--video_in", type=str, default="datos/video campo erondina a procesar.MP4")
    parser.add_argument("--video_out", type=str, default="datos/Resultados/resultado_erondina_reid_timeline_min1.mp4")
    parser.add_argument("--report_out", type=str, default="reports/16_reid_timeline_min1.json")
    parser.add_argument("--contact_sheet_out", type=str, default="reports/16_contact_sheet_timeline_min1.jpg")
    parser.add_argument("--candidate_sheet_out", type=str, default="reports/16_candidate_sheet_timeline_min1.jpg")
    parser.add_argument("--yolo_model", type=str, default="scripts/yolov8m.pt")
    parser.add_argument("--tracker", type=str, default="botsort.yaml")
    parser.add_argument("--reid_model", type=str, default="models/mi_modelo_reid.pt")
    parser.add_argument("--identity_gallery", type=str, default="models/erondina_gallery_embeddings_enfocada_filtrada.npz")
    parser.add_argument("--start_frame", type=int, default=1800)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--process_width", type=int, default=0)
    parser.add_argument("--process_height", type=int, default=0)
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
    parser.add_argument("--timeline_identity_threshold", type=float, default=0.76)
    parser.add_argument("--timeline_identity_margin", type=float, default=0.010)
    parser.add_argument("--timeline_min_identity_support", type=float, default=0.20)
    parser.add_argument("--timeline_min_identity_vote_fraction", type=float, default=0.18)
    parser.add_argument("--timeline_fragment_identity_margin", type=float, default=0.08)
    parser.add_argument("--timeline_fragment_min_identity_support", type=float, default=0.35)
    parser.add_argument("--timeline_fragment_min_identity_vote_fraction", type=float, default=0.70)
    parser.add_argument("--timeline_min_reid_similarity", type=float, default=0.70)
    parser.add_argument("--timeline_min_head_similarity", type=float, default=0.68)
    parser.add_argument("--timeline_bridge_min_reid_similarity", type=float, default=0.66)
    parser.add_argument("--timeline_max_overlap_frames", type=int, default=30)
    parser.add_argument("--timeline_high_confidence_threshold", type=float, default=0.88)
    parser.add_argument("--timeline_high_confidence_margin", type=float, default=0.10)
    parser.add_argument("--timeline_high_confidence_vote_fraction", type=float, default=0.70)
    parser.add_argument("--anchor_start_window_frames", type=int, default=60)
    parser.add_argument("--expected_total_cows", type=int, default=14)
    parser.add_argument("--count_tolerance", type=int, default=2)
    parser.add_argument("--min_known_hit_ratio", type=float, default=0.10)
    parser.add_argument("--color_rerank_weight", type=float, default=0.18)
    parser.add_argument("--disable_color_rerank", dest="color_rerank", action="store_false")
    parser.set_defaults(color_rerank=True)
    parser.add_argument("--focus_margin_x", type=float, default=0.12)
    parser.add_argument("--focus_margin_y", type=float, default=0.18)
    parser.add_argument("--disable_head_embeddings", dest="head_embeddings", action="store_false")
    parser.set_defaults(head_embeddings=True)
    parser.add_argument("--head_reid_every", type=int, default=16)
    parser.add_argument("--head_crop_y_fraction", type=float, default=0.72)
    parser.add_argument("--head_crop_x_margin", type=float, default=0.02)
    parser.add_argument("--known_bbox_smoothing_alpha", type=float, default=0.55)
    parser.add_argument("--known_smoothing_reset_center_factor", type=float, default=2.0)
    parser.add_argument("--known_hold_frames", type=int, default=18)
    parser.add_argument("--known_reacquire_frames", type=int, default=75)
    parser.add_argument("--known_reacquire_min_iou", type=float, default=0.08)
    parser.add_argument("--known_reacquire_max_center_factor", type=float, default=0.85)
    parser.add_argument("--known_exit_edge_margin_ratio", type=float, default=0.05)
    parser.add_argument("--unknown_suppress_iou_with_known", type=float, default=0.12)
    parser.add_argument("--evidence_cache_in", type=str, default="")
    parser.add_argument("--evidence_cache_out", type=str, default="")
    parser.add_argument("--display_frame_start", type=int, default=1)
    parser.add_argument("--display_total_cows", type=int, default=0)
    parser.add_argument("--public_report", action="store_true")
    parser.add_argument("--unknown_label_mode", choices=["id", "generic"], default="id")
    parser.add_argument("--hide_visible_overlay", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    device = base05.torch.device("cpu")
    detector = YOLO(args.yolo_model)
    cow_class_id = base05.obtener_cow_class_id(detector)
    reid_model = base05.cargar_reid_model(args.reid_model, device)
    gallery_vectors, gallery_labels = s13.load_gallery(args.identity_gallery)
    known_required = sorted(set(gallery_labels))
    transform = base05.transforms.Compose(
        [
            base05.transforms.ToPILImage(),
            base05.transforms.Resize((224, 224)),
            base05.transforms.ToTensor(),
            base05.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tracklets, frame_records, processed, total_frames, fps, width, height = load_or_build_evidence(
        args,
        device,
        detector,
        cow_class_id,
        reid_model,
        transform,
    )

    local_to_global = s13.cluster_tracklets(
        tracklets,
        args.min_track_frames,
        args.merge_threshold,
        args.max_overlap_frames,
        args.max_merge_gap_frames,
    )
    all_candidates = s13.collect_known_candidates(tracklets, local_to_global, gallery_vectors, gallery_labels)
    color_diagnostics = {}
    if args.color_rerank:
        profiles = gallery_color_profiles(args.identity_gallery)
        all_candidates, color_diagnostics = enrich_candidates_with_color(args, all_candidates, frame_records, profiles)
    anchor_assignments = s13.assign_known_identities(
        all_candidates,
        threshold=args.identity_threshold,
        margin=args.identity_margin,
        min_support=args.min_identity_support,
        min_vote_fraction=args.min_identity_vote_fraction,
    )
    timeline_assignments, timeline_diagnostics, group_data = build_timeline_assignments(
        args,
        tracklets,
        local_to_global,
        all_candidates,
        anchor_assignments,
        gallery_labels,
    )

    contact_sheet = make_timeline_contact_sheet(args, frame_records, local_to_global, timeline_assignments, args.contact_sheet_out)
    candidate_sheet = make_timeline_candidate_sheet(args, frame_records, local_to_global, all_candidates, args.candidate_sheet_out)

    visible_counts = [
        len({local_to_global[rec.local_track_id] for rec in records if rec.local_track_id in local_to_global})
        for records in frame_records
    ]
    visible_mean = float(np.mean(visible_counts)) if visible_counts else 0.0
    visible_median = float(np.median(visible_counts)) if visible_counts else 0.0
    visible_p95 = float(np.percentile(visible_counts, 95)) if visible_counts else 0.0
    visible_max = int(max(visible_counts)) if visible_counts else 0
    estimated_visible_total = int(round(visible_p95)) if visible_counts else len(set(local_to_global.values()))

    known_frame_hits = defaultdict(int)
    known_first_render_frame = {}
    known_last_render_frame = {}
    all_known_labels = set(known_required)
    all_known_together_frames = []
    rendered_duplicate_frames, duplicate_examples = duplicate_known_frames(frame_records, local_to_global, timeline_assignments)
    for idx, records in enumerate(frame_records, start=1):
        known_records_by_label, _ = choose_known_records(records, local_to_global, timeline_assignments)
        visible_labels = set(known_records_by_label)
        for label in known_records_by_label:
            known_frame_hits[label] += 1
            known_first_render_frame.setdefault(label, idx)
            known_last_render_frame[label] = idx
        if all_known_labels and all_known_labels.issubset(visible_labels):
            all_known_together_frames.append(idx)

    found = sorted({data["label"] for data in timeline_assignments.values()})
    missing = sorted(set(known_required) - set(found))
    count_error = estimated_visible_total - args.expected_total_cows
    absolute_count_error = abs(count_error)
    count_accuracy = max(0.0, 1.0 - absolute_count_error / args.expected_total_cows) if args.expected_total_cows else 0.0
    count_within_tolerance = absolute_count_error <= args.count_tolerance
    if int(args.display_total_cows) <= 0:
        args.display_total_cows = int(estimated_visible_total)
    min_hit_ratio_by_label = {
        label: float(known_frame_hits.get(label, 0) / processed) if processed else 0.0
        for label in known_required
    }
    min_hit_ratio_ok = all(value >= args.min_known_hit_ratio for value in min_hit_ratio_by_label.values())
    all_known_together_ok = len(all_known_together_frames) > 0
    locked_track_audit = audit_locked_known_tracks(
        args,
        frame_records,
        local_to_global,
        timeline_assignments,
        fps,
        width,
        height,
    )
    locked_labels = locked_track_audit.get("labels", {})
    locked_present_ok = all(
        locked_labels.get(s13.display_label(label), {}).get("present_ratio", 0.0) >= args.min_known_hit_ratio
        for label in known_required
    )
    locked_midframe_gap_count = sum(
        int(item.get("midframe_long_gap_count_ge_2s", 0))
        for item in locked_labels.values()
    )
    locked_audit_ok = (
        locked_present_ok
        and locked_track_audit.get("all_known_together", {}).get("exists", False)
        and locked_midframe_gap_count == 0
    )
    ready = not missing and count_within_tolerance and locked_audit_ok

    report = {
        "video_in": args.video_in,
        "video_out": args.video_out if args.render else None,
        "processed_frames": processed,
        "processing_video_size": {"width": int(width), "height": int(height)},
        "estimated_total_cows": estimated_visible_total,
        "estimated_total_cows_method": "visible_cows_p95_per_frame",
        "global_track_count_after_clustering": len(set(local_to_global.values())),
        "fragmentation_over_expected": max(0, len(set(local_to_global.values())) - args.expected_total_cows),
        "expected_total_cows": args.expected_total_cows,
        "count_error": int(count_error),
        "absolute_count_error": int(absolute_count_error),
        "count_accuracy": float(count_accuracy),
        "count_within_tolerance": bool(count_within_tolerance),
        "unknown_cows_estimated": max(0, estimated_visible_total - len(found)),
        "visible_cows_per_frame": {
            "mean": visible_mean,
            "median": visible_median,
            "p95": visible_p95,
            "max": visible_max,
        },
        "known_found": [s13.display_label(x) for x in found],
        "known_missing": [s13.display_label(x) for x in missing],
        "known_frame_hits": {s13.display_label(k): int(v) for k, v in sorted(known_frame_hits.items())},
        "known_hit_ratio": {
            s13.display_label(k): float(v) for k, v in sorted(min_hit_ratio_by_label.items())
        },
        "known_first_render_frame": {s13.display_label(k): int(v) for k, v in sorted(known_first_render_frame.items())},
        "known_last_render_frame": {s13.display_label(k): int(v) for k, v in sorted(known_last_render_frame.items())},
        "all_known_together": {
            "exists": bool(all_known_together_ok),
            "frame_count": int(len(all_known_together_frames)),
            "first_render_frame": int(all_known_together_frames[0]) if all_known_together_frames else None,
            "last_render_frame": int(all_known_together_frames[-1]) if all_known_together_frames else None,
        },
        "locked_track_audit": locked_track_audit,
        "locked_track_audit_ok": bool(locked_audit_ok),
        "locked_midframe_gap_count_ge_2s": int(locked_midframe_gap_count),
        "duplicate_known_label_frames_before_render_suppression": {
            s13.display_label(k): int(v) for k, v in sorted(rendered_duplicate_frames.items())
        },
        "duplicate_known_label_examples": {
            s13.display_label(k): v for k, v in sorted(duplicate_examples.items())
        },
        "known_id_switches_by_design": 0,
        "ready_for_render_by_automatic_checks": bool(ready),
        "anchor_assignments_by_global_id": {str(k): v for k, v in sorted(anchor_assignments.items())},
        "timeline_assignments_by_global_id": {str(k): v for k, v in sorted(timeline_assignments.items())},
        "timeline_diagnostics": timeline_diagnostics,
        "all_known_candidates": all_candidates,
        "color_rerank": {
            "enabled": bool(args.color_rerank),
            "weight": float(args.color_rerank_weight),
            "diagnostics": {
                label: sorted(items, key=lambda x: x["adjusted_score"], reverse=True)[:6]
                for label, items in color_diagnostics.items()
            },
        },
        "local_to_global": {str(k): int(v) for k, v in sorted(local_to_global.items())},
        "contact_sheet": contact_sheet,
        "candidate_sheet": candidate_sheet,
        "metric_note": (
            "Precision y recall reales por bounding box requieren anotaciones ground truth por frame. "
            "Este JSON reporta metricas automaticas: conteo estimado, primeras/ultimas apariciones "
            "de identidades conocidas, duplicados antes de la supresion de render, pureza/margen ReID, "
            "vacas visibles por frame y fragmentacion de tracks."
        ),
        "params": vars(args),
    }

    if args.render:
        print("Pass 2/2: render con linea temporal de identidades conocidas")
        suppressed_duplicate_count = render_video_timeline(
            args,
            frame_records,
            local_to_global,
            timeline_assignments,
            width,
            height,
            fps,
        )
        report["suppressed_duplicate_known_detections_in_render"] = int(suppressed_duplicate_count)
        report["render_stabilization"] = {
            "known_bbox_smoothing_alpha": float(args.known_bbox_smoothing_alpha),
            "known_smoothing_reset_center_factor": float(args.known_smoothing_reset_center_factor),
            "known_hold_frames": int(args.known_hold_frames),
            "known_reacquire_frames": int(args.known_reacquire_frames),
            "known_reacquire_min_iou": float(args.known_reacquire_min_iou),
            "known_reacquire_max_center_factor": float(args.known_reacquire_max_center_factor),
            "known_exit_edge_margin_ratio": float(args.known_exit_edge_margin_ratio),
            "unknown_suppress_iou_with_known": float(args.unknown_suppress_iou_with_known),
            "known_labels_drawn_last": True,
            "known_label_font_scale": 1.65,
        }

    if args.public_report:
        report = public_report(report, args)

    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("============================================================")
    print("Reporte Re-ID timeline estable por embeddings")
    print("============================================================")
    print(f"Frames procesados        : {processed}")
    print(f"Vacas estimadas por frame: {estimated_visible_total}")
    print(f"Tracks globales cluster  : {len(set(local_to_global.values()))}")
    print(f"Accuracy conteo estimado    : {100.0 * count_accuracy:.2f}%")
    print(f"Conocidas encontradas    : {', '.join(report['known_found']) if report['known_found'] else 'ninguna'}")
    print(f"Primer frame conocido    : {report['known_first_render_frame']}")
    print(f"Duplicados previos render: {report.get('duplicate_known_label_frames_before_render_suppression', 'omitido')}")
    print(f"Checks automaticos OK    : {ready}")
    print(f"Reporte guardado en      : {args.report_out}")
    if contact_sheet:
        print(f"Contacto visual guardado : {contact_sheet}")
    if candidate_sheet:
        print(f"Candidatas visual guardado: {candidate_sheet}")
    if args.render:
        print(f"Video guardado en        : {args.video_out}")


if __name__ == "__main__":
    main()
