#!/usr/bin/env python3
"""CowTrack local web interface.

This app intentionally uses only the Python standard library. It provides a
small local web server that launches the existing CowTrack pipeline, streams
progress, summarizes the final JSON report, and can notify a Telegram chat.
"""

from __future__ import annotations

import cgi
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
STATIC_DIR = APP_DIR / "static"
UPLOAD_DIR = APP_DIR / "uploads"
RUNS_DIR = APP_DIR / "runs"
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "16_reid_timeline_erondina.py"

DEFAULT_VIDEO = "/Volumes/T7/cow-tracker-mvp/datos/Resultado final/archivo a procesar.mp4"
DEFAULT_RESULT_DIR = "/Volumes/T7/cow-tracker-mvp/datos/Resultado final"
DEFAULT_PIPELINE_PYTHON = "/Volumes/T7/cow-tracker-mvp/.venv/bin/python3"


def resolve_pipeline_python() -> str:
    candidates = [
        os.getenv("COWTRACK_PYTHON", ""),
        DEFAULT_PIPELINE_PYTHON,
        str(REPO_ROOT / ".venv" / "bin" / "python3"),
        sys.executable,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return sys.executable


@dataclass
class JobState:
    run_id: str | None = None
    status: str = "idle"
    step: str = "Esperando video"
    progress: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    command: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    report: dict | None = None
    artifacts: dict = field(default_factory=dict)
    error: str | None = None
    returncode: int | None = None


STATE = JobState()
STATE_LOCK = threading.Lock()


def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def text_response(handler: BaseHTTPRequestHandler, text: str, status: int = 200) -> None:
    raw = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def read_form(handler: BaseHTTPRequestHandler) -> dict:
    content_type = handler.headers.get("Content-Type", "")
    length = int(handler.headers.get("Content-Length", "0"))
    if "multipart/form-data" in content_type:
        form = cgi.FieldStorage(
            fp=handler.rfile,
            headers=handler.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(length),
            },
        )
        data: dict[str, str] = {}
        upload = form["video_file"] if "video_file" in form else None
        if upload is not None and getattr(upload, "filename", ""):
            safe_name = Path(upload.filename).name
            dest = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
            with dest.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
            data["video_path"] = str(dest)
        for key in form.keys():
            if key == "video_file":
                continue
            item = form[key]
            data[key] = item.value if hasattr(item, "value") else str(item)
        return data

    raw = handler.rfile.read(length).decode("utf-8")
    if "application/json" in content_type:
        return json.loads(raw or "{}")
    return {k: v[0] for k, v in urllib.parse.parse_qs(raw).items()}


def clamp_int(value: str | int | None, default: int, minimum: int = 0) -> int:
    try:
        parsed = int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def clamp_float(value: str | float | None, default: float, minimum: float = 0.0) -> float:
    try:
        parsed = float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def summarize_report(report: dict | None) -> dict:
    if not report:
        return {}
    known = report.get("known_found", [])
    expected = report.get("expected_total_cows")
    estimated = report.get("estimated_total_cows")
    accuracy = report.get("count_accuracy")
    scores = report.get("identity_scores", {})
    return {
        "processed_frames": report.get("processed_frames"),
        "estimated_total_cows": estimated,
        "expected_total_cows": expected,
        "count_error": report.get("count_error"),
        "count_accuracy": accuracy,
        "count_accuracy_percent": round(float(accuracy or 0) * 100, 2),
        "known_found": known,
        "known_missing": report.get("known_missing", []),
        "known_id_switches_by_design": report.get("known_id_switches_by_design"),
        "locked_track_audit_ok": report.get("locked_track_audit_ok"),
        "identity_scores": {k: round(float(v), 4) for k, v in scores.items()},
    }


def telegram_message(summary: dict, artifacts: dict) -> str:
    known = ", ".join(summary.get("known_found") or ["sin identidades"])
    missing = ", ".join(summary.get("known_missing") or ["ninguna"])
    video = artifacts.get("video_path", "no disponible")
    report = artifacts.get("report_path", "no disponible")
    return (
        "CowTrack finalizó el procesamiento.\n\n"
        f"Conteo estimado: {summary.get('estimated_total_cows')} vacas\n"
        f"Referencia configurada: {summary.get('expected_total_cows')} vacas\n"
        f"Accuracy de conteo: {summary.get('count_accuracy_percent')}%\n"
        f"Re-ID encontradas: {known}\n"
        f"Re-ID faltantes: {missing}\n"
        f"ID switches conocidos: {summary.get('known_id_switches_by_design')}\n\n"
        f"Video: {video}\n"
        f"Reporte JSON: {report}"
    )


def send_telegram(text: str, token: str, chat_id: str) -> tuple[bool, str]:
    if not token or not chat_id:
        return False, "Falta COWTRACK_TELEGRAM_BOT_TOKEN o COWTRACK_TELEGRAM_CHAT_ID."
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    try:
        with urllib.request.urlopen(url, data=payload, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        return bool(data.get("ok")), data.get("description", "Mensaje enviado")
    except Exception as exc:  # pragma: no cover - depends on network
        return False, str(exc)


def build_command(form: dict, run_dir: Path) -> tuple[list[str], dict]:
    video_path = form.get("video_path") or DEFAULT_VIDEO
    video_path = str(Path(video_path).expanduser())
    if not Path(video_path).exists():
        raise FileNotFoundError(f"No se encontró el video: {video_path}")

    output_name = form.get("output_name") or "RESULTADO_COWTRACK.mp4"
    if not output_name.lower().endswith(".mp4"):
        output_name += ".mp4"

    result_dir = Path(form.get("result_dir") or DEFAULT_RESULT_DIR).expanduser()
    try:
        result_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        result_dir = run_dir

    video_out = result_dir / output_name
    report_out = run_dir / "cowtrack_report.json"
    contact_sheet = run_dir / "cowtrack_contact_sheet.jpg"
    candidate_sheet = run_dir / "cowtrack_candidate_sheet.jpg"
    evidence_cache = run_dir / "cowtrack_evidence.pkl"

    expected_total = clamp_int(form.get("expected_total_cows"), 13, 1)
    start_frame = clamp_int(form.get("start_frame"), 0, 0)
    max_frames = clamp_int(form.get("max_frames"), 0, 0)
    process_width = clamp_int(form.get("process_width"), 1920, 0)
    process_height = clamp_int(form.get("process_height"), 1080, 0)
    det_conf = clamp_float(form.get("det_conf"), 0.18, 0.01)

    cmd = [
        resolve_pipeline_python(),
        "-u",
        str(PIPELINE_SCRIPT),
        "--video_in",
        video_path,
        "--video_out",
        str(video_out),
        "--report_out",
        str(report_out),
        "--contact_sheet_out",
        str(contact_sheet),
        "--candidate_sheet_out",
        str(candidate_sheet),
        "--evidence_cache_out",
        str(evidence_cache),
        "--start_frame",
        str(start_frame),
        "--expected_total_cows",
        str(expected_total),
        "--display_total_cows",
        str(expected_total),
        "--display_frame_start",
        "0",
        "--process_width",
        str(process_width),
        "--process_height",
        str(process_height),
        "--det_conf",
        str(det_conf),
        "--unknown_label_mode",
        "generic",
        "--hide_visible_overlay",
        "--public_report",
        "--render",
    ]
    if max_frames > 0:
        cmd.extend(["--max_frames", str(max_frames)])

    artifacts = {
        "run_dir": str(run_dir),
        "video_path": str(video_out),
        "report_path": str(report_out),
        "contact_sheet_path": str(contact_sheet),
        "candidate_sheet_path": str(candidate_sheet),
        "evidence_cache_path": str(evidence_cache),
    }
    return cmd, artifacts


def update_state(**kwargs) -> None:
    with STATE_LOCK:
        for key, value in kwargs.items():
            setattr(STATE, key, value)


def append_log(line: str) -> None:
    clean = line.rstrip()
    if not clean:
        return
    with STATE_LOCK:
        STATE.logs.append(clean)
        STATE.logs = STATE.logs[-240:]
        track_match = re.search(r"Track frame\s+(\d+)/(\d+)", clean)
        render_match = re.search(r"Render frame\s+(\d+)/(\d+)", clean)
        if track_match:
            current = int(track_match.group(1))
            total = max(1, int(track_match.group(2)))
            STATE.step = clean
            STATE.progress = min(62, max(8, int(8 + 54 * current / total)))
        elif render_match:
            current = int(render_match.group(1))
            total = max(1, int(render_match.group(2)))
            STATE.step = clean
            STATE.progress = min(94, max(65, int(65 + 29 * current / total)))
        elif "Pass 2/2" in clean:
            STATE.step = "Renderizando video final"
            STATE.progress = max(65, STATE.progress)
        elif "Reporte guardado" in clean:
            STATE.step = "Guardando reporte"
            STATE.progress = max(94, STATE.progress)


def run_pipeline(form: dict, run_id: str) -> None:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        cmd, artifacts = build_command(form, run_dir)
        update_state(
            status="running",
            step="Inicializando CowTrack",
            progress=5,
            command=cmd,
            artifacts=artifacts,
            error=None,
            returncode=None,
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            append_log(line)
        returncode = process.wait()
        report = None
        report_path = Path(artifacts["report_path"])
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        summary = summarize_report(report)
        if returncode == 0:
            update_state(
                status="completed",
                step="Proceso finalizado",
                progress=100,
                finished_at=time.time(),
                report=report,
                returncode=returncode,
            )
            if form.get("telegram_enabled") == "true":
                token = form.get("telegram_token") or os.getenv("COWTRACK_TELEGRAM_BOT_TOKEN", "")
                chat_id = form.get("telegram_chat_id") or os.getenv("COWTRACK_TELEGRAM_CHAT_ID", "")
                ok, message = send_telegram(telegram_message(summary, artifacts), token, chat_id)
                append_log(f"Telegram: {'OK' if ok else 'ERROR'} - {message}")
        else:
            update_state(
                status="failed",
                step="El pipeline terminó con error",
                progress=100,
                finished_at=time.time(),
                report=report,
                error=f"Código de salida: {returncode}",
                returncode=returncode,
            )
    except Exception as exc:
        append_log(f"ERROR: {exc}")
        update_state(
            status="failed",
            step="No se pudo ejecutar el pipeline",
            progress=100,
            finished_at=time.time(),
            error=str(exc),
        )


def state_payload() -> dict:
    with STATE_LOCK:
        report_summary = summarize_report(STATE.report)
        return {
            "run_id": STATE.run_id,
            "status": STATE.status,
            "step": STATE.step,
            "progress": STATE.progress,
            "started_at": STATE.started_at,
            "finished_at": STATE.finished_at,
            "logs": STATE.logs,
            "summary": report_summary,
            "artifacts": STATE.artifacts,
            "error": STATE.error,
            "returncode": STATE.returncode,
        }


class CowTrackHandler(BaseHTTPRequestHandler):
    server_version = "CowTrackWeb/1.0"

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/":
            target = STATIC_DIR / "index.html"
        elif path.startswith("/static/"):
            target = STATIC_DIR / path.removeprefix("/static/")
        elif path.startswith("/runs/"):
            target = RUNS_DIR / path.removeprefix("/runs/")
        else:
            self.send_response(404)
            self.end_headers()
            return
        if target.exists() and target.is_file():
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
            self.send_header("Content-Length", str(target.stat().st_size))
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/":
            self.serve_static(STATIC_DIR / "index.html")
        elif path == "/api/config":
            json_response(
                self,
                {
                    "default_video": DEFAULT_VIDEO,
                    "default_result_dir": DEFAULT_RESULT_DIR,
                    "telegram_ready": bool(
                        os.getenv("COWTRACK_TELEGRAM_BOT_TOKEN")
                        and os.getenv("COWTRACK_TELEGRAM_CHAT_ID")
                    ),
                },
            )
        elif path == "/api/status":
            json_response(self, state_payload())
        elif path == "/api/report":
            with STATE_LOCK:
                json_response(self, STATE.report or {})
        elif path.startswith("/static/"):
            self.serve_static(STATIC_DIR / path.removeprefix("/static/"))
        elif path.startswith("/runs/"):
            self.serve_static(RUNS_DIR / path.removeprefix("/runs/"))
        else:
            text_response(self, "No encontrado", 404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/run":
            form = read_form(self)
            with STATE_LOCK:
                if STATE.status == "running":
                    json_response(self, {"ok": False, "error": "Ya hay un proceso en ejecución."}, 409)
                    return
                run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
                STATE.run_id = run_id
                STATE.status = "queued"
                STATE.step = "Preparando ejecución"
                STATE.progress = 1
                STATE.started_at = time.time()
                STATE.finished_at = None
                STATE.logs = []
                STATE.report = None
                STATE.artifacts = {}
                STATE.error = None
                STATE.returncode = None
            worker = threading.Thread(target=run_pipeline, args=(form, run_id), daemon=True)
            worker.start()
            json_response(self, {"ok": True, "run_id": run_id})
        elif parsed.path == "/api/telegram/test":
            form = read_form(self)
            token = form.get("telegram_token") or os.getenv("COWTRACK_TELEGRAM_BOT_TOKEN", "")
            chat_id = form.get("telegram_chat_id") or os.getenv("COWTRACK_TELEGRAM_CHAT_ID", "")
            ok, message = send_telegram("CowTrack: prueba de notificación configurada correctamente.", token, chat_id)
            json_response(self, {"ok": ok, "message": message}, 200 if ok else 400)
        else:
            text_response(self, "No encontrado", 404)

    def serve_static(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            allowed_roots = [STATIC_DIR.resolve(), RUNS_DIR.resolve()]
            if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
                text_response(self, "Acceso no permitido", 403)
                return
            if not resolved.exists() or not resolved.is_file():
                text_response(self, "No encontrado", 404)
                return
            content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
            raw = resolved.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except Exception as exc:
            text_response(self, str(exc), 500)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[CowTrack] {self.address_string()} - {fmt % args}")


def main() -> None:
    ensure_dirs()
    port = int(os.getenv("COWTRACK_PORT", "7860"))
    server = ThreadingHTTPServer(("127.0.0.1", port), CowTrackHandler)
    print(f"CowTrack Web listo en http://127.0.0.1:{port}")
    print("Presiona Ctrl+C para detener.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
