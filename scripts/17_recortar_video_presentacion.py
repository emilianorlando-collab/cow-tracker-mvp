#!/usr/bin/env python3
"""Create the CowTrack presentation source clip from the original field video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_timecode(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Timecode invalido: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recorta el clip base de presentacion CowTrack.")
    parser.add_argument("--video_in", required=True)
    parser.add_argument("--video_out", required=True)
    parser.add_argument("--start", default="03:50")
    parser.add_argument("--duration", type=float, default=47.0)
    parser.add_argument("--output_width", type=int, default=0)
    parser.add_argument("--output_height", type=int, default=0)
    args = parser.parse_args()

    video_in = Path(args.video_in)
    video_out = Path(args.video_out)
    if not video_in.exists():
        raise FileNotFoundError(video_in)

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = args.output_width or source_width
    height = args.output_height or source_height
    start_frame = int(round(parse_timecode(args.start) * fps))
    frame_count = int(round(args.duration * fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame >= total_frames:
        raise ValueError("El frame inicial queda fuera del video.")

    video_out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear el video de salida: {video_out}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    while written < frame_count:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1
        if written == 1 or written % 100 == 0:
            print(f"Frame recortado {written}/{frame_count}")

    cap.release()
    writer.release()

    print("============================================================")
    print("Clip CowTrack generado")
    print("============================================================")
    print(f"Entrada       : {video_in}")
    print(f"Salida        : {video_out}")
    print(f"Inicio        : {args.start}")
    print(f"Duracion      : {written / fps:.2f} s")
    print(f"FPS           : {fps:.3f}")
    print(f"Frames        : {written}")
    print(f"Resolucion src: {source_width}x{source_height}")
    print(f"Resolucion out: {width}x{height}")


if __name__ == "__main__":
    main()
