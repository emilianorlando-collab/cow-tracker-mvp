#!/usr/bin/env python3
"""CowTrack product mockup with local persistence and real pipeline hook."""

from __future__ import annotations

import cgi
import html
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
MOCKUP_DIR = REPO_ROOT / "mockup"
STATIC_DIR = MOCKUP_DIR / "static"
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "16_reid_timeline_erondina.py"

T7_ROOT = Path("/Volumes/T7/cow-tracker-mvp")
T7_MOCKUP = T7_ROOT / "mockup"
USER_DATA_DIR = T7_MOCKUP / "user_data"
REPORTS_MOCKUP_DIR = T7_MOCKUP / "reports_mockup"
UPLOADS_DIR = T7_MOCKUP / "uploads"
RUNS_DIR = T7_MOCKUP / "runs"
FAST_DEMO_VIDEO = T7_ROOT / "datos" / "Resultado final" / "archivo a procesar demo 5s.mp4"
FULL_DEMO_VIDEO = T7_ROOT / "datos" / "Resultado final" / "archivo a procesar.mp4"
REAL_REPORT = REPO_ROOT / "app" / "runs" / "20260617_201441_cad968e6" / "cowtrack_report.json"
REAL_CONTACT = REPO_ROOT / "app" / "runs" / "20260617_201441_cad968e6" / "cowtrack_contact_sheet.jpg"
REAL_VIDEO = T7_ROOT / "datos" / "Resultado final" / "RESULTADO_COWTRACK.mp4"
DEFAULT_PIPELINE_PYTHON = T7_ROOT / ".venv" / "bin" / "python3"
ADMIN_COVER_SOURCES = {
    "Marta": T7_ROOT / "datos" / "erondina_reid" / "Marta" / "galeria" / "Captura de pantalla 2026-06-01 a las 2.47.26 p. m..png",
}

USERS = {"admin": {"password": "admin", "name": "admin", "role": "Administrador"}}
SESSIONS: dict[str, str] = {}


@dataclass
class JobState:
    run_id: str | None = None
    username: str | None = None
    status: str = "idle"
    step: str = "Listo para iniciar"
    progress: int = 0
    logs: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    report: dict | None = None
    artifacts: dict = field(default_factory=dict)
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None


STATE = JobState()
STATE_LOCK = threading.Lock()
ACTIVE_PROCESS: subprocess.Popen | None = None
ACTIVE_PROCESS_LOCK = threading.Lock()


def pipeline_python() -> str:
    env_python = os.getenv("COWTRACK_PYTHON")
    for candidate in [env_python, str(DEFAULT_PIPELINE_PYTHON), sys.executable]:
        if candidate and Path(candidate).exists():
            return candidate
    return sys.executable


def ensure_dirs() -> None:
    for path in [T7_MOCKUP, USER_DATA_DIR, REPORTS_MOCKUP_DIR, UPLOADS_DIR, RUNS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    seed_admin_data()


def copy_if_exists(src: Path, dest: Path) -> str | None:
    if not src.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)
    return str(dest)


def seed_admin_data() -> None:
    admin_dir = USER_DATA_DIR / "admin"
    catalog_dir = admin_dir / "catalog"
    admin_dir.mkdir(parents=True, exist_ok=True)
    catalog_dir.mkdir(parents=True, exist_ok=True)

    cows = [
        ("Margarita", "Vaca castaña", "Identidad catalogada para seguimiento individual."),
        ("Maria", "Vaca negra", "Identidad catalogada para seguimiento individual."),
        ("Marta", "Vaca castaña", "Identidad catalogada para seguimiento individual."),
    ]
    for name, phenotype, note in cows:
        cow_dir = catalog_dir / name
        cow_dir.mkdir(parents=True, exist_ok=True)
        cover_source = ADMIN_COVER_SOURCES.get(name)
        if cover_source and cover_source.exists():
            shutil.copy2(cover_source, cow_dir / "cover.png")
        metadata = {
            "name": name,
            "status": "catalogada",
            "phenotype": phenotype,
            "note": note,
            "embedding_status": "disponible",
            "created_at": "2026-06-17",
        }
        meta_path = cow_dir / "metadata.json"
        current = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        current.update(metadata)
        meta_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")

    report_dir = REPORTS_MOCKUP_DIR / "admin"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = copy_if_exists(REAL_REPORT, report_dir / "reporte_real_cowtrack_47s.json")
    contact_path = copy_if_exists(REAL_CONTACT, report_dir / "reporte_real_contact_sheet.jpg")
    video_path = str(REAL_VIDEO) if REAL_VIDEO.exists() else ""
    history_path = report_dir / "historial_reportes.json"
    if report_path and not history_path.exists():
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        history = [
            build_user_report_entry(
                "Reporte inicial CowTrack",
                report,
                {
                    "video_path": video_path,
                    "report_path": report_path,
                    "contact_sheet_path": contact_path or "",
                },
            )
        ]
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


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


def parse_cookies(handler: BaseHTTPRequestHandler) -> dict[str, str]:
    raw = handler.headers.get("Cookie", "")
    cookies = {}
    for part in raw.split(";"):
        if "=" in part:
            key, value = part.strip().split("=", 1)
            cookies[key] = value
    return cookies


def current_user(handler: BaseHTTPRequestHandler) -> str | None:
    token = parse_cookies(handler).get("cowtrack_session")
    return SESSIONS.get(token or "")


def require_user(handler: BaseHTTPRequestHandler) -> str | None:
    username = current_user(handler)
    if not username:
        json_response(handler, {"ok": False, "error": "Iniciá sesión para continuar."}, 401)
        return None
    return username


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
        data: dict[str, object] = {}
        for key in form.keys():
            item = form[key]
            if isinstance(item, list):
                data[key] = item
            elif getattr(item, "filename", ""):
                data[key] = item
            else:
                data[key] = item.value
        return data
    raw = handler.rfile.read(length).decode("utf-8")
    if "application/json" in content_type:
        return json.loads(raw or "{}")
    return {k: v[0] for k, v in urllib.parse.parse_qs(raw).items()}


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9ÁÉÍÓÚáéíóúÑñ _-]+", "", value).strip()
    return cleaned or "Sin nombre"


def save_upload_file(field, dest_dir: Path, fallback_name: str) -> Path | None:
    if field is None or not getattr(field, "filename", ""):
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = Path(field.filename or fallback_name).name
    dest = dest_dir / f"{int(time.time())}_{name}"
    with dest.open("wb") as out:
        shutil.copyfileobj(field.file, out)
    return dest


def catalog(username: str) -> list[dict]:
    out = []
    for cow_dir in sorted((USER_DATA_DIR / username / "catalog").glob("*")):
        if not cow_dir.is_dir():
            continue
        meta_path = cow_dir / "metadata.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        cover = cow_dir / "cover.png"
        if not cover.exists():
            cover = cow_dir / "cover.jpg"
        photos = list(cow_dir.glob("*.jpg")) + list(cow_dir.glob("*.png")) + list(cow_dir.glob("*.jpeg"))
        out.append(
            {
                "name": metadata.get("name", cow_dir.name),
                "status": metadata.get("status", "catalogada"),
                "phenotype": metadata.get("phenotype", "Identidad bovina"),
                "embedding_status": metadata.get("embedding_status", "pendiente"),
                "photo_count": len(photos),
                "cover_url": f"/mockup-files/user_data/{username}/catalog/{urllib.parse.quote(cow_dir.name)}/{urllib.parse.quote(cover.name)}"
                if cover.exists()
                else "/static/favicon.svg",
            }
        )
    return out


def reports_history(username: str) -> list[dict]:
    path = REPORTS_MOCKUP_DIR / username / "historial_reportes.json"
    if not path.exists():
        return []
    return [enrich_report_entry(entry) for entry in json.loads(path.read_text(encoding="utf-8"))]


def save_reports_history(username: str, history: list[dict]) -> None:
    path = REPORTS_MOCKUP_DIR / username / "historial_reportes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def build_user_report_entry(title: str, report: dict, artifacts: dict) -> dict:
    found = report.get("known_found", [])
    accuracy = float(report.get("count_accuracy") or 0)
    estimated = report.get("estimated_total_cows")
    expected = report.get("expected_total_cows")
    entry = {
        "id": uuid.uuid4().hex[:10],
        "title": title,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "estimated_total_cows": estimated,
        "expected_total_cows": expected,
        "count_accuracy_percent": round(accuracy * 100, 2),
        "reidentified_cows": found,
        "unknown_cows_estimated": report.get("unknown_cows_estimated"),
        "healthy_message": "Rodeo contado y vacas catalogadas localizadas.",
        "video_path": artifacts.get("video_path", ""),
        "report_path": artifacts.get("report_path", ""),
        "contact_sheet_path": artifacts.get("contact_sheet_path", ""),
        "summary": {
            "frames": report.get("processed_frames"),
            "id_switches": report.get("known_id_switches_by_design"),
            "tracking_ok": report.get("locked_track_audit_ok"),
        },
    }
    return enrich_report_entry(entry, report)


def artifact_exists(path: str | None) -> bool:
    return bool(path) and Path(path).exists()


def load_technical_report(entry: dict) -> dict:
    report_path = entry.get("report_path")
    if not report_path or not Path(report_path).exists():
        return {}
    try:
        return json.loads(Path(report_path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def enrich_report_entry(entry: dict, technical: dict | None = None) -> dict:
    technical = technical or load_technical_report(entry)
    estimated = entry.get("estimated_total_cows")
    expected = entry.get("expected_total_cows")
    accuracy = float(entry.get("count_accuracy_percent") or 0)
    found = entry.get("reidentified_cows") or []
    missing = technical.get("known_missing", []) if technical else []
    unknown = entry.get("unknown_cows_estimated")
    hit_ratio = technical.get("known_hit_ratio", {}) if technical else {}
    identity_scores = technical.get("identity_scores", {}) if technical else {}
    valid_count = bool(expected and estimated and accuracy >= 80)
    valid_reid = (
        len(found) >= 3
        and not missing
        and all(float(hit_ratio.get(name, 0.0)) >= 0.10 for name in ["Marta", "Maria", "Margarita"])
        and all(float(identity_scores.get(name, 0.0)) >= 0.80 for name in ["Marta", "Maria", "Margarita"])
        and bool(technical.get("locked_track_audit_ok", entry.get("summary", {}).get("tracking_ok")))
    )
    no_detections = bool(expected and int(estimated or 0) == 0)
    if no_detections:
        entry["status_label"] = "Sin detecciones"
        entry["status_tone"] = "bad"
    elif valid_count and valid_reid:
        entry["status_label"] = "Validado"
        entry["status_tone"] = "ok"
    else:
        entry["status_label"] = "Validación parcial"
        entry["status_tone"] = "warn"
    entry["count_delta"] = None if estimated is None or expected is None else int(estimated) - int(expected)
    entry["reidentified_count"] = len(found)
    entry["unknown_cows_estimated"] = unknown if unknown is not None else max(0, int(estimated or 0) - len(found))
    entry["artifact_status"] = {
        "video": artifact_exists(entry.get("video_path")),
        "contact_sheet": artifact_exists(entry.get("contact_sheet_path")),
        "report": artifact_exists(entry.get("report_path")),
    }
    entry["technical"] = {
        "processed_frames": technical.get("processed_frames", entry.get("summary", {}).get("frames")),
        "identity_scores": technical.get("identity_scores", {}),
        "known_hit_ratio": technical.get("known_hit_ratio", {}),
        "known_missing": missing,
        "id_switches": technical.get("known_id_switches_by_design", entry.get("summary", {}).get("id_switches")),
        "tracking_ok": technical.get("locked_track_audit_ok", entry.get("summary", {}).get("tracking_ok")),
        "ready_for_render": technical.get("ready_for_render_by_automatic_checks"),
        "count_within_tolerance": technical.get("count_within_tolerance"),
        "per_frame_detection_summary": technical.get("per_frame_detection_summary", {}),
        "processing_video_size": technical.get("processing_video_size", {}),
        "metric_note": technical.get("metric_note", ""),
    }
    return entry


def summarize_state_report(report: dict | None) -> dict:
    if not report:
        return {}
    return build_user_report_entry("Conteo diario CowTrack", report, {})


def telegram_text(entry: dict) -> str:
    cows = ", ".join(entry.get("reidentified_cows") or ["sin vacas catalogadas"])
    return (
        "CowTrack - Reporte diario\n\n"
        f"Conteo del rodeo: {entry.get('estimated_total_cows')} vacas\n"
        f"Referencia ingresada: {entry.get('expected_total_cows')} vacas\n"
        f"Confiabilidad de conteo: {entry.get('count_accuracy_percent')}%\n"
        f"Vacas catalogadas localizadas: {cows}\n\n"
        "El video y las capturas quedaron disponibles en el panel de CowTrack."
    )


def send_telegram(text: str, token: str, chat_id: str) -> tuple[bool, str]:
    if not token or not chat_id:
        return False, "Configurá token y chat ID."
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    try:
        with urllib.request.urlopen(url, data=payload, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        return bool(data.get("ok")), data.get("description", "Mensaje enviado")
    except Exception as exc:
        return False, str(exc)


def update_state(**kwargs) -> None:
    with STATE_LOCK:
        for key, value in kwargs.items():
            setattr(STATE, key, value)


def reset_state(terminate_process: bool = True) -> None:
    global ACTIVE_PROCESS
    if terminate_process:
        with ACTIVE_PROCESS_LOCK:
            process = ACTIVE_PROCESS
            ACTIVE_PROCESS = None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    update_state(
        run_id=None,
        username=None,
        status="idle",
        step="Listo para iniciar",
        progress=0,
        logs=[],
        summary={},
        report=None,
        artifacts={},
        error=None,
        started_at=None,
        finished_at=None,
    )


def append_log(line: str) -> None:
    clean = line.rstrip()
    if not clean:
        return
    with STATE_LOCK:
        STATE.logs.append(clean)
        STATE.logs = STATE.logs[-120:]
        track = re.search(r"Track frame\s+(\d+)/(\d+)", clean)
        render = re.search(r"Render frame\s+(\d+)/(\d+)", clean)
        if track:
            current, total = int(track.group(1)), max(1, int(track.group(2)))
            STATE.progress = min(62, max(8, int(8 + 54 * current / total)))
            STATE.step = "Analizando el rodeo"
        elif render:
            current, total = int(render.group(1)), max(1, int(render.group(2)))
            STATE.progress = min(94, max(65, int(65 + 29 * current / total)))
            STATE.step = "Preparando el video final"
        elif "Pass 2/2" in clean:
            STATE.step = "Generando resultado visual"
            STATE.progress = max(65, STATE.progress)


def build_command(form: dict, username: str, run_dir: Path) -> tuple[list[str], dict]:
    video_path = str(form.get("video_path") or "").strip()
    upload = form.get("video_file")
    if upload is not None and getattr(upload, "filename", ""):
        saved = save_upload_file(upload, UPLOADS_DIR / username / "videos", "video.mp4")
        video_path = str(saved)
        if "demo 5s" in Path(saved).name.lower() and FULL_DEMO_VIDEO.exists():
            video_path = str(FULL_DEMO_VIDEO)
    fast_demo = form.get("use_fast_demo") == "true"
    if fast_demo and FAST_DEMO_VIDEO.exists():
        video_path = str(FAST_DEMO_VIDEO)
    if not video_path:
        raise ValueError("Seleccioná un video para procesar.")
    if not Path(video_path).exists():
        raise FileNotFoundError(f"No se encontró el video: {video_path}")

    expected_total = int(form.get("expected_total_cows") or 13)
    output_name = safe_name(str(form.get("output_name") or "resultado_cowtrack")) + ".mp4"
    video_out = RUNS_DIR / username / run_dir.name / output_name
    report_out = run_dir / "reporte_tecnico.json"
    contact_sheet = run_dir / "capturas_reidentificacion.jpg"
    candidate_sheet = run_dir / "candidatos_tecnicos.jpg"
    evidence_cache = run_dir / "evidencia.pkl"
    max_frames = int(form.get("max_frames") or 0)

    cmd = [
        pipeline_python(),
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
        "0",
        "--expected_total_cows",
        str(expected_total),
        "--display_total_cows",
        str(expected_total),
        "--display_frame_start",
        "0",
        "--process_width",
        "1920",
        "--process_height",
        "1080",
        "--det_conf",
        "0.18",
        "--unknown_label_mode",
        "generic",
        "--hide_visible_overlay",
        "--public_report",
        "--min_track_frames",
        "5",
        "--reid_every",
        "4",
        "--head_reid_every",
        "8",
        "--min_known_hit_ratio",
        "0.10",
        "--count_tolerance",
        str(max(2, round(expected_total * 0.20))),
        "--render",
    ]
    if max_frames > 0:
        cmd.extend(["--max_frames", str(max_frames)])
    if fast_demo:
        cmd.extend(["--max_frames", "150"])
    artifacts = {
        "video_path": str(video_out),
        "report_path": str(report_out),
        "contact_sheet_path": str(contact_sheet),
        "run_dir": str(run_dir),
    }
    return cmd, artifacts


def run_pipeline(form: dict, username: str, run_id: str) -> None:
    global ACTIVE_PROCESS
    run_dir = RUNS_DIR / username / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        cmd, artifacts = build_command(form, username, run_dir)
        update_state(
            run_id=run_id,
            username=username,
            status="running",
            step="Preparando el análisis",
            progress=4,
            logs=[],
            summary={},
            report=None,
            artifacts=artifacts,
            error=None,
            started_at=time.time(),
            finished_at=None,
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
        with ACTIVE_PROCESS_LOCK:
            ACTIVE_PROCESS = process
        assert process.stdout is not None
        for line in process.stdout:
            append_log(line)
        returncode = process.wait()
        with ACTIVE_PROCESS_LOCK:
            if ACTIVE_PROCESS is process:
                ACTIVE_PROCESS = None
        report = None
        if Path(artifacts["report_path"]).exists():
            report = json.loads(Path(artifacts["report_path"]).read_text(encoding="utf-8"))
        if returncode != 0:
            raise RuntimeError(f"El procesamiento terminó con código {returncode}")
        entry = build_user_report_entry("Conteo diario CowTrack", report or {}, artifacts)
        history = reports_history(username)
        history.insert(0, entry)
        save_reports_history(username, history[:20])
        update_state(
            status="completed",
            step="Reporte listo",
            progress=100,
            finished_at=time.time(),
            report=report,
            summary=entry,
        )
    except Exception as exc:
        with ACTIVE_PROCESS_LOCK:
            ACTIVE_PROCESS = None
        append_log(f"ERROR: {exc}")
        update_state(status="failed", step="No se pudo completar el análisis", progress=100, error=str(exc), finished_at=time.time())


def state_payload() -> dict:
    with STATE_LOCK:
        return {
            "run_id": STATE.run_id,
            "status": STATE.status,
            "step": STATE.step,
            "progress": STATE.progress,
            "summary": STATE.summary,
            "artifacts": STATE.artifacts,
            "error": STATE.error,
            "started_at": STATE.started_at,
            "finished_at": STATE.finished_at,
        }


def find_report(username: str, report_id: str) -> dict | None:
    for entry in reports_history(username):
        if entry.get("id") == report_id:
            return entry
    return None


def pdf_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace("\r", " ").replace("\n", " ")


def report_filename(entry: dict) -> str:
    raw = str(entry.get("date") or time.strftime("%Y-%m-%d %H:%M"))
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", raw).strip("_")
    return f"cowtrack_{cleaned or time.strftime('%Y%m%d_%H%M')}.pdf"


def make_report_pdf(entry: dict) -> bytes:
    found = ", ".join(entry.get("reidentified_cows") or ["sin vacas catalogadas"])
    tech = entry.get("technical", {})
    visible = tech.get("per_frame_detection_summary", {})
    size = tech.get("processing_video_size", {})

    def wrap(text: str, max_chars: int = 86) -> list[str]:
        words = str(text).split()
        rows: list[str] = []
        current: list[str] = []
        for word in words:
            if len(" ".join(current + [word])) > max_chars and current:
                rows.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            rows.append(" ".join(current))
        return rows or [""]

    stream_lines: list[str] = []

    def color(hex_color: str) -> str:
        hex_color = hex_color.lstrip("#")
        parts = [int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        return f"{parts[0]:.3f} {parts[1]:.3f} {parts[2]:.3f}"

    def rect(x: int, y: int, w: int, h: int, fill: str) -> None:
        stream_lines.append(f"q {color(fill)} rg {x} {y} {w} {h} re f Q")

    def text(x: int, y: int, value: str, size: int = 10, font: str = "F1", fill: str = "#18211d") -> None:
        stream_lines.append(f"BT /{font} {size} Tf {color(fill)} rg 1 0 0 1 {x} {y} Tm ({pdf_escape(value)}) Tj ET")

    def wrapped_text(x: int, y: int, value: str, max_chars: int = 86, size: int = 10, leading: int = 14, fill: str = "#4f5d55") -> int:
        for line in wrap(value, max_chars):
            text(x, y, line, size=size, fill=fill)
            y -= leading
        return y

    rect(0, 782, 612, 60, "#f2f7f1")
    rect(46, 768, 38, 38, "#36a856")
    text(57, 779, "C", 22, "F2", "#ffffff")
    stream_lines.append(f"q {color('#66d978')} rg 49 809 19 8 re f Q")
    stream_lines.append(f"q {color('#2ba657')} rg 66 812 22 8 re f Q")
    text(96, 783, "CowTrack", 25, "F2", "#18211d")
    text(96, 768, "Informe ejecutivo de conteo y reidentificación ganadera", 9, "F1", "#6f766f")

    text(46, 730, entry.get("title", "Conteo diario CowTrack"), 20, "F2", "#18211d")
    text(46, 708, f"Fecha: {entry.get('date', '-')}", 10, "F1", "#6f766f")
    rect(46, 677, 520, 1, "#dde4dc")

    card_y = 626
    cards = [
        ("Conteo obtenido", f"{entry.get('estimated_total_cows', '-')} vacas"),
        ("Referencia", f"{entry.get('expected_total_cows', '-')} vacas"),
        ("Confiabilidad", f"{entry.get('count_accuracy_percent', '-')}%"),
        ("Estado", str(entry.get("status_label", "-"))),
    ]
    for idx, (label, value) in enumerate(cards):
        x = 46 + idx * 130
        rect(x, card_y, 118, 58, "#f7faf6")
        text(x + 10, card_y + 38, label, 8, "F1", "#6f766f")
        text(x + 10, card_y + 17, value, 13, "F2", "#18211d")

    y = 560
    text(46, y, "Resumen operativo", 14, "F2", "#18211d")
    y -= 22
    summary = (
        "El análisis consolida el conteo del rodeo, separa vacas catalogadas de vacas no catalogadas y "
        "presenta evidencia visual para apoyar decisiones de stock ganadero."
    )
    y = wrapped_text(46, y, summary, 90, 10, 15)
    y -= 12
    text(46, y, f"Diferencia contra referencia: {entry.get('count_delta', '-')}", 10, "F1", "#4f5d55")
    y -= 16
    text(46, y, f"Vacas no catalogadas estimadas: {entry.get('unknown_cows_estimated', '-')}", 10, "F1", "#4f5d55")

    y -= 34
    text(46, y, "Reidentificación", 14, "F2", "#18211d")
    y -= 22
    y = wrapped_text(46, y, f"Vacas catalogadas localizadas: {found}.", 86, 10, 15)
    y -= 10
    text(46, y, f"Seguimiento temporal validado: {'sí' if tech.get('tracking_ok') else 'no'}", 10, "F1", "#4f5d55")
    y -= 16
    text(46, y, f"ID switches reportados: {tech.get('id_switches', '-')}", 10, "F1", "#4f5d55")

    y -= 34
    text(46, y, "Lectura técnica", 14, "F2", "#18211d")
    y -= 22
    text(46, y, f"Frames procesados: {tech.get('processed_frames', '-')}", 10, "F1", "#4f5d55")
    y -= 16
    text(46, y, f"Resolución procesada: {size.get('width', '-')} x {size.get('height', '-')}", 10, "F1", "#4f5d55")
    y -= 16
    text(46, y, f"Detecciones visibles promedio: {visible.get('mean_visible_detections', '-')}", 10, "F1", "#4f5d55")

    y -= 34
    text(46, y, "Conclusión", 14, "F2", "#18211d")
    y -= 22
    conclusion = (
        "El reporte permite revisar el conteo, confirmar las vacas reconocidas y conservar un "
        "registro visual del análisis diario para seguimiento del rodeo."
    )
    wrapped_text(46, y, conclusion, 90, 10, 15)

    rect(46, 40, 520, 1, "#dde4dc")
    text(46, 24, "CowTrack · Inteligencia artificial aplicada al control de stock ganadero", 8, "F1", "#6f766f")

    stream = "\n".join(stream_lines).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    return bytes(pdf)


def bytes_response(handler: BaseHTTPRequestHandler, raw: bytes, content_type: str, filename: str | None = None) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(raw)))
    if filename:
        handler.send_header("Content-Disposition", f'inline; filename="{filename}"')
    handler.end_headers()
    handler.wfile.write(raw)


class Handler(BaseHTTPRequestHandler):
    server_version = "CowTrackMockup/2.0"

    def do_HEAD(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/":
            self.serve_file(STATIC_DIR / "index.html")
        elif path.startswith("/static/"):
            self.serve_file(STATIC_DIR / path.removeprefix("/static/"))
        elif path == "/favicon.ico":
            self.serve_file(STATIC_DIR / "favicon.svg")
        elif path.startswith("/mockup-files/"):
            self.serve_file(T7_MOCKUP / urllib.parse.unquote(path.removeprefix("/mockup-files/")))
        elif path.startswith("/cowtrack-files/"):
            self.serve_file(T7_ROOT / urllib.parse.unquote(path.removeprefix("/cowtrack-files/")))
        elif path == "/api/session":
            username = current_user(self)
            json_response(self, {"authenticated": bool(username), "user": username, "profile": USERS.get(username or "", {})})
        elif path == "/api/dashboard":
            username = require_user(self)
            if username:
                json_response(self, {"catalog": catalog(username), "reports": reports_history(username), "state": state_payload()})
        elif path == "/api/status":
            json_response(self, state_payload())
        elif path == "/api/report_pdf":
            username = require_user(self)
            if username:
                query = urllib.parse.parse_qs(parsed.query)
                entry = find_report(username, str((query.get("id") or [""])[0]))
                if not entry:
                    json_response(self, {"ok": False, "error": "No se encontró el reporte."}, 404)
                    return
                bytes_response(self, make_report_pdf(entry), "application/pdf", report_filename(entry))
        else:
            text_response(self, "No encontrado", 404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/api/login":
                reset_state(terminate_process=True)
                form = read_form(self)
                username = str(form.get("username", ""))
                password = str(form.get("password", ""))
                if USERS.get(username, {}).get("password") != password:
                    json_response(self, {"ok": False, "error": "Usuario o contraseña incorrectos."}, 401)
                    return
                token = uuid.uuid4().hex
                SESSIONS[token] = username
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Set-Cookie", f"cowtrack_session={token}; Path=/; SameSite=Lax")
                raw = json.dumps({"ok": True, "user": username}).encode("utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif parsed.path == "/api/register":
                form = read_form(self)
                username = safe_name(str(form.get("username") or "")).replace(" ", "_")
                password = str(form.get("password") or "")
                name = safe_name(str(form.get("name") or username or "Usuario CowTrack"))
                if not username or not password:
                    json_response(self, {"ok": False, "error": "Completá usuario y contraseña."}, 400)
                    return
                if username in USERS:
                    json_response(self, {"ok": False, "error": "Ese usuario ya existe."}, 409)
                    return
                USERS[username] = {"password": password, "name": name, "role": "Productor"}
                (USER_DATA_DIR / username / "catalog").mkdir(parents=True, exist_ok=True)
                (REPORTS_MOCKUP_DIR / username).mkdir(parents=True, exist_ok=True)
                token = uuid.uuid4().hex
                SESSIONS[token] = username
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Set-Cookie", f"cowtrack_session={token}; Path=/; SameSite=Lax")
                raw = json.dumps({"ok": True, "user": username}).encode("utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif parsed.path == "/api/social_login":
                reset_state(terminate_process=True)
                form = read_form(self)
                provider = safe_name(str(form.get("provider") or "Social")).replace(" ", "_")
                username = f"{provider.lower()}_demo"
                USERS.setdefault(username, {"password": "", "name": f"Usuario {provider}", "role": "Productor"})
                (USER_DATA_DIR / username / "catalog").mkdir(parents=True, exist_ok=True)
                (REPORTS_MOCKUP_DIR / username).mkdir(parents=True, exist_ok=True)
                token = uuid.uuid4().hex
                SESSIONS[token] = username
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Set-Cookie", f"cowtrack_session={token}; Path=/; SameSite=Lax")
                raw = json.dumps({"ok": True, "user": username, "provider": provider}).encode("utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif parsed.path == "/api/logout":
                reset_state(terminate_process=True)
                token = parse_cookies(self).get("cowtrack_session")
                if token:
                    SESSIONS.pop(token, None)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Set-Cookie", "cowtrack_session=; Path=/; Max-Age=0; SameSite=Lax")
                raw = b'{"ok": true}'
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif parsed.path == "/api/catalog":
                username = require_user(self)
                if not username:
                    return
                form = read_form(self)
                name = safe_name(str(form.get("cow_name") or "Nueva vaca"))
                cow_dir = USER_DATA_DIR / username / "catalog" / name
                cow_dir.mkdir(parents=True, exist_ok=True)
                files = form.get("cow_photos")
                items = files if isinstance(files, list) else [files]
                saved_count = 0
                for item in items:
                    saved = save_upload_file(item, cow_dir, "foto.jpg")
                    if saved:
                        saved_count += 1
                        cover = cow_dir / "cover.jpg"
                        if not cover.exists():
                            shutil.copy2(saved, cover)
                metadata = {
                    "name": name,
                    "status": "catalogada",
                    "phenotype": str(form.get("phenotype") or "Identidad bovina"),
                    "embedding_status": "disponible" if saved_count else "pendiente",
                    "photos": saved_count,
                    "created_at": time.strftime("%Y-%m-%d"),
                }
                (cow_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                json_response(self, {"ok": True, "catalog": catalog(username)})
            elif parsed.path == "/api/catalog/delete":
                username = require_user(self)
                if not username:
                    return
                form = read_form(self)
                name = safe_name(str(form.get("cow_name") or ""))
                target = USER_DATA_DIR / username / "catalog" / name
                if target.exists():
                    shutil.rmtree(target)
                json_response(self, {"ok": True, "catalog": catalog(username)})
            elif parsed.path == "/api/run":
                username = require_user(self)
                if not username:
                    return
                with STATE_LOCK:
                    if STATE.status == "running":
                        json_response(self, {"ok": False, "error": "Ya hay un conteo en proceso."}, 409)
                        return
                form = read_form(self)
                run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
                update_state(
                    run_id=run_id,
                    username=username,
                    status="queued",
                    step="Preparando el análisis",
                    progress=1,
                    logs=[],
                    summary={},
                    report=None,
                    artifacts={},
                    error=None,
                    started_at=time.time(),
                    finished_at=None,
                )
                threading.Thread(target=run_pipeline, args=(form, username, run_id), daemon=True).start()
                json_response(self, {"ok": True, "run_id": run_id})
            elif parsed.path == "/api/reset":
                reset_state(terminate_process=True)
                json_response(self, {"ok": True, "state": state_payload()})
            elif parsed.path == "/api/telegram":
                username = require_user(self)
                if not username:
                    return
                form = read_form(self)
                history = reports_history(username)
                entry = history[0] if history else state_payload().get("summary", {})
                ok, message = send_telegram(
                    telegram_text(entry),
                    str(form.get("telegram_token") or os.getenv("COWTRACK_TELEGRAM_BOT_TOKEN", "")),
                    str(form.get("telegram_chat_id") or os.getenv("COWTRACK_TELEGRAM_CHAT_ID", "")),
                )
                json_response(self, {"ok": ok, "message": message}, 200 if ok else 400)
            elif parsed.path == "/api/contact":
                form = read_form(self)
                dest = T7_MOCKUP / "contactos"
                dest.mkdir(parents=True, exist_ok=True)
                item = {"date": time.strftime("%Y-%m-%d %H:%M"), **{k: str(v) for k, v in form.items()}}
                (dest / f"contacto_{int(time.time())}.json").write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
                json_response(self, {"ok": True, "message": "Solicitud recibida. El equipo CowTrack se contactará a la brevedad."})
            else:
                text_response(self, "No encontrado", 404)
        except Exception as exc:
            json_response(self, {"ok": False, "error": str(exc)}, 500)

    def serve_file(self, path: Path) -> None:
        resolved = path.resolve()
        allowed = [STATIC_DIR.resolve(), T7_MOCKUP.resolve(), T7_ROOT.resolve()]
        if not any(str(resolved).startswith(str(root)) for root in allowed):
            text_response(self, "Acceso no permitido", 403)
            return
        if not resolved.exists() or not resolved.is_file():
            text_response(self, "No encontrado", 404)
            return
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        file_size = resolved.stat().st_size
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            start_s, _, end_s = range_header.removeprefix("bytes=").partition("-")
            start = int(start_s or 0)
            end = int(end_s) if end_s else file_size - 1
            end = min(end, file_size - 1)
            if start > end or start >= file_size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with resolved.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    remaining -= len(chunk)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        with resolved.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break

    def log_message(self, fmt: str, *args) -> None:
        print(f"[CowTrack Mockup] {fmt % args}")


def main() -> None:
    ensure_dirs()
    port = int(os.getenv("COWTRACK_PORT", "7860"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"CowTrack mockup listo en http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
