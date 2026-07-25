"""Docker-ready browser interface for the 9-ball foul detector.

The original foul-detection engine is intentionally reused from
Main_Camera_Code.py.  This module replaces only the Windows/Tkinter UI and
DirectShow camera access:

1. The browser obtains permission to use the Windows camera.
2. MediaRecorder records a clip in the browser.
3. The clip is uploaded to this FastAPI service.
4. The original YOLO/foul-processing loop analyzes the saved clip.
5. Annotated frames, foul events, and output-file links are sent back to the
   browser.

Run locally:
    python Programs/Docker_Ready_Camera_Code.py

Then open:
    http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import uvicorn
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse
from ultralytics import YOLO

import Main_Camera_Code as legacy


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_ROOT = PROJECT_ROOT / "Trained_Models"
OUTPUT_ROOT = Path(
    os.environ.get(
        "OUTPUT_DIR",
        str(PROJECT_ROOT / "Output_Folder" / "Docker_Web"),
    )
).resolve()

HOST = os.environ.get("APP_HOST", "0.0.0.0")
PORT = int(os.environ.get("APP_PORT", "8000"))
JPEG_QUALITY = max(40, min(95, int(os.environ.get("JPEG_QUALITY", "82"))))
MAX_UPLOAD_MB = max(1, int(os.environ.get("MAX_UPLOAD_MB", "2048")))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

ALLOWED_VIDEO_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}
TERMINAL_JOB_STATES = {"completed", "stopped", "error"}


class _NullRoot:
    """Minimal Tk-compatible object used by the unchanged legacy engine."""

    def title(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def geometry(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def configure(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def after(self, _delay_ms: int, callback=None, *args: Any) -> None:
        if callback is not None:
            callback(*args)


class _NullListbox:
    """Receives legacy listbox updates without creating a desktop window."""

    def delete(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def insert(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def see(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@dataclass
class DetectionJob:
    job_id: str
    model_name: str
    output_dir: Path
    input_path: Path
    fps_override: Optional[float] = None
    status: str = "uploading"
    message: str = "Receiving video"
    error: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_source_frames: int = 0
    processed_source_frames: int = 0
    frame_sequence: int = 0
    latest_jpeg: Optional[bytes] = None
    events: list[dict[str, Any]] = field(default_factory=list)
    engine: Optional["BrowserFoulDetector"] = None
    stop_requested: bool = False
    paused: bool = False
    version: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def update(self, **changes: Any) -> None:
        with self.lock:
            for name, value in changes.items():
                setattr(self, name, value)
            self.version += 1

    def add_event(self, event_type: str, message: str, **extra: Any) -> None:
        with self.lock:
            self.events.append(
                {
                    "type": event_type,
                    "message": message,
                    "time": datetime.now(timezone.utc).isoformat(),
                    **extra,
                }
            )
            self.version += 1

    def set_frame(
        self,
        jpeg: bytes,
        processed_source_frames: int,
        total_source_frames: int,
    ) -> None:
        with self.lock:
            self.latest_jpeg = jpeg
            self.frame_sequence += 1
            self.processed_source_frames = processed_source_frames
            self.total_source_frames = total_source_frames
            self.version += 1

    def output_files(self) -> list[str]:
        if not self.output_dir.exists():
            return []
        return sorted(
            path.name
            for path in self.output_dir.iterdir()
            if path.is_file() and path.resolve() != self.input_path.resolve()
        )

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            total = self.total_source_frames
            current = self.processed_source_frames
            progress = (
                min(100.0, max(0.0, current * 100.0 / total))
                if total > 0
                else 0.0
            )
            return {
                "type": "status",
                "job_id": self.job_id,
                "model": self.model_name,
                "status": self.status,
                "message": self.message,
                "error": self.error,
                "created_at": self.created_at,
                "progress": round(progress, 1),
                "processed_source_frames": current,
                "total_source_frames": total,
                "frame_sequence": self.frame_sequence,
                "paused": self.paused,
                "outputs": self.output_files()
                if self.status in TERMINAL_JOB_STATES
                else [],
                "version": self.version,
            }


def discover_models() -> dict[str, Path]:
    """Return only model folders containing a usable .pt weights file."""

    models: dict[str, Path] = {}
    if not MODELS_ROOT.is_dir():
        return models

    for model_dir in sorted(MODELS_ROOT.iterdir(), key=lambda p: p.name.lower()):
        if not model_dir.is_dir():
            continue
        candidate = Path(legacy._model_pt_path(model_dir.name)).resolve()
        if candidate.is_file() and candidate.suffix.lower() == ".pt":
            models[model_dir.name] = candidate
    return models


class BrowserFoulDetector(legacy.FoulDetectorApp):
    """Headless adapter around the original foul-detection implementation."""

    def _build_ui(self) -> None:
        self.foul_listbox = _NullListbox()

    def _populate_models(self) -> None:
        return None

    def refresh_cameras(self) -> None:
        return None

    def __init__(
        self,
        job: DetectionJob,
        model_path: Path,
    ) -> None:
        self.job = job
        self._cleaned_up = False
        super().__init__(_NullRoot())
        self.output_dir = str(job.output_dir)
        self.model_path = str(model_path)

        self.job.update(status="loading_model", message=f"Loading {job.model_name}")
        self.model = YOLO(self.model_path)
        self.job.add_event("system", f"Loaded model {job.model_name}")

    def run(self) -> None:
        """Open the uploaded recording and run the unchanged processing loop."""

        video_path = str(self.job.input_path)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(
                "The uploaded video could not be opened. "
                "Use MP4, AVI, MOV, MKV, or a browser-generated WebM recording."
            )

        fps_override = self.job.fps_override
        if fps_override is not None and 1.0 <= fps_override <= 240.0:
            source_fps = float(fps_override)
        else:
            source_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if not 1.0 <= source_fps <= 240.0:
            source_fps = 30.0

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("The uploaded video has an invalid frame size.")

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._frame_stride = max(1, int(round(source_fps / 30.0)))
        self._src_frame_no = 0
        self._src_fps = source_fps
        self._last_annotated = None
        self.fps = source_fps / self._frame_stride

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        annotated_path = self.job.output_dir / f"annotated_{timestamp}.mp4"
        writer, codec = self._make_h264_writer(
            str(annotated_path),
            self._src_fps,
            (width, height),
        )
        if writer is None:
            raise RuntimeError("Could not create the annotated output video.")
        self.video_writer = writer

        self.foul_log = (self.job.output_dir / f"foul_log_{timestamp}.txt").open(
            "w",
            encoding="utf-8",
        )
        self.foul_log.write(
            f"Foul Log - {timestamp}\n"
            f"Source: {self.job.input_path.name}\n"
            f"Model: {self.job.model_name}\n\n"
        )

        self.debug_log = (
            self.job.output_dir / f"debug_positions_{timestamp}.txt"
        ).open("w", encoding="utf-8")
        self.debug_log.write(
            f"Ball Position Debug Log - {timestamp}\n"
            f"Source: {self.job.input_path.name}\n"
            f"Model: {self.job.model_name}\n\n"
        )

        self._reset_state()
        self.is_playing = True
        self.is_paused = False
        self.job.update(
            status="processing",
            message=(
                f"Analyzing {width}x{height} at {self.fps:.1f} effective FPS "
                f"(output codec: {codec})"
            ),
            total_source_frames=total_frames,
        )
        self.job.add_event(
            "system",
            f"Analysis started at {self.fps:.1f} effective FPS",
        )

        legacy.FoulDetectorApp._process_loop(self)

    def _add_event_log(self, message: str) -> None:
        legacy.FoulDetectorApp._add_event_log(self, message)
        event_text = self.event_log[-1] if self.event_log else message
        self.job.add_event("event", event_text)

    def _register_foul(self, reason: str) -> None:
        legacy.FoulDetectorApp._register_foul(self, reason)
        elapsed = self.frame_idx / max(self.fps, 1e-6)
        timestamp = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        self.job.add_event(
            "foul",
            reason,
            video_time=timestamp,
            foul_number=self._foul_count,
        )

    def _display_frame(self, frame_bgr) -> None:
        height, width = frame_bgr.shape[:2]
        max_width = 1200
        if width > max_width:
            scale = max_width / width
            frame_bgr = cv2.resize(
                frame_bgr,
                (max_width, max(1, int(height * scale))),
            )

        ok, encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        if not ok:
            return

        self.job.set_frame(
            encoded.tobytes(),
            processed_source_frames=self._src_frame_no,
            total_source_frames=self.job.total_source_frames,
        )

    def _cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.foul_log is not None:
            self.foul_log.close()
            self.foul_log = None
        if self.debug_log is not None:
            self.debug_log.close()
            self.debug_log = None

        if self.job.error:
            return
        if self.job.stop_requested:
            self.job.update(status="stopped", message="Analysis stopped")
            self.job.add_event("system", "Analysis stopped")
        else:
            self.job.update(
                status="completed",
                message=f"Analysis complete - {self._foul_count} foul(s) detected",
                processed_source_frames=max(
                    self.job.processed_source_frames,
                    self.job.total_source_frames,
                ),
            )
            self.job.add_event(
                "system",
                f"Analysis complete - {self._foul_count} foul(s) detected",
            )


class JobManager:
    """Owns jobs and limits model processing to one recording at a time."""

    def __init__(self) -> None:
        self.jobs: dict[str, DetectionJob] = {}
        self.active_job_id: Optional[str] = None
        self.lock = threading.RLock()

    def reserve(
        self,
        model_name: str,
        suffix: str,
        fps_override: Optional[float],
    ) -> DetectionJob:
        with self.lock:
            if self.active_job_id:
                active = self.jobs.get(self.active_job_id)
                if active and active.status not in TERMINAL_JOB_STATES:
                    raise HTTPException(
                        status_code=409,
                        detail="Another recording is currently being processed.",
                    )

            job_id = uuid.uuid4().hex
            output_dir = (OUTPUT_ROOT / job_id).resolve()
            output_dir.mkdir(parents=True, exist_ok=False)
            input_path = output_dir / f"source{suffix}"
            job = DetectionJob(
                job_id=job_id,
                model_name=model_name,
                output_dir=output_dir,
                input_path=input_path,
                fps_override=fps_override,
            )
            self.jobs[job_id] = job
            self.active_job_id = job_id
            return job

    def get(self, job_id: str) -> DetectionJob:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job

    def release(self, job: DetectionJob) -> None:
        with self.lock:
            if self.active_job_id == job.job_id:
                self.active_job_id = None

    def start(self, job: DetectionJob, model_path: Path) -> None:
        worker = threading.Thread(
            target=self._run,
            args=(job, model_path),
            daemon=True,
            name=f"detector-{job.job_id[:8]}",
        )
        worker.start()

    def _run(self, job: DetectionJob, model_path: Path) -> None:
        try:
            engine = BrowserFoulDetector(job=job, model_path=model_path)
            job.engine = engine
            if job.stop_requested:
                engine._cleanup()
                return
            engine.run()
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            job.update(status="error", message="Analysis failed", error=message)
            job.add_event("error", message)
            if job.engine is not None:
                job.engine._cleanup()
        finally:
            self.release(job)


OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
manager = JobManager()
app = FastAPI(
    title="9-Ball Foul Detector",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX_HTML


@app.get("/healthz")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
async def list_models() -> dict[str, Any]:
    models = discover_models()
    return {
        "models": list(models),
        "default": "YOLO11n" if "YOLO11n" in models else next(iter(models), None),
    }


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    model: str = Form(...),
    fps: Optional[float] = Form(None),
) -> dict[str, Any]:
    models = discover_models()
    if model not in models:
        raise HTTPException(status_code=400, detail="Unknown model selection.")

    original_name = Path(file.filename or "recording.webm").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format.",
        )

    normalized_fps = None
    if fps is not None:
        if not 1.0 <= fps <= 240.0:
            raise HTTPException(status_code=400, detail="FPS must be between 1 and 240.")
        normalized_fps = float(fps)

    job = manager.reserve(model, suffix, normalized_fps)
    received = 0
    try:
        with job.input_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                received += len(chunk)
                if received > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video exceeds the {MAX_UPLOAD_MB} MB upload limit.",
                    )
                destination.write(chunk)
        await file.close()

        if received == 0:
            raise HTTPException(status_code=400, detail="The uploaded video is empty.")

        job.update(
            status="queued",
            message=f"Upload complete ({received / (1024 * 1024):.1f} MB)",
        )
        job.add_event("system", f"Received {original_name}")
        manager.start(job, models[model])
        return {"job_id": job.job_id, "status": job.status}
    except Exception:
        manager.release(job)
        if job.status not in TERMINAL_JOB_STATES:
            job.update(status="error", message="Upload failed")
        raise


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    return manager.get(job_id).snapshot()


@app.post("/api/jobs/{job_id}/pause")
async def pause_job(job_id: str) -> dict[str, Any]:
    job = manager.get(job_id)
    if job.engine is None or job.status not in {"processing", "paused"}:
        raise HTTPException(status_code=409, detail="This job cannot be paused.")
    job.engine.is_paused = True
    job.update(status="paused", message="Analysis paused", paused=True)
    return job.snapshot()


@app.post("/api/jobs/{job_id}/resume")
async def resume_job(job_id: str) -> dict[str, Any]:
    job = manager.get(job_id)
    if job.engine is None or job.status != "paused":
        raise HTTPException(status_code=409, detail="This job is not paused.")
    job.engine.is_paused = False
    job.update(status="processing", message="Analysis resumed", paused=False)
    return job.snapshot()


@app.post("/api/jobs/{job_id}/stop")
async def stop_job(job_id: str) -> dict[str, Any]:
    job = manager.get(job_id)
    if job.status in TERMINAL_JOB_STATES:
        return job.snapshot()
    job.stop_requested = True
    if job.engine is not None:
        job.engine.is_paused = False
        job.engine.is_playing = False
    job.update(message="Stopping analysis", paused=False)
    return job.snapshot()


@app.get("/api/jobs/{job_id}/files/{filename}")
async def download_output(job_id: str, filename: str) -> FileResponse:
    job = manager.get(job_id)
    safe_name = Path(filename).name
    target = (job.output_dir / safe_name).resolve()
    if target.parent != job.output_dir.resolve() or not target.is_file():
        raise HTTPException(status_code=404, detail="Output file not found.")
    return FileResponse(target, filename=safe_name)


@app.websocket("/ws/jobs/{job_id}")
async def stream_job(websocket: WebSocket, job_id: str) -> None:
    try:
        job = manager.get(job_id)
    except HTTPException:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    last_frame_sequence = -1
    last_event_index = 0
    last_version = -1
    terminal_seen_at: Optional[float] = None

    try:
        while True:
            with job.lock:
                snapshot = job.snapshot()
                current_version = job.version
                frame_sequence = job.frame_sequence
                jpeg = job.latest_jpeg
                pending_events = list(job.events[last_event_index:])
                last_event_index = len(job.events)

            if current_version != last_version:
                await websocket.send_text(json.dumps(snapshot))
                last_version = current_version

            for event in pending_events:
                await websocket.send_text(json.dumps(event))

            if jpeg is not None and frame_sequence != last_frame_sequence:
                await websocket.send_bytes(jpeg)
                last_frame_sequence = frame_sequence

            if snapshot["status"] in TERMINAL_JOB_STATES:
                if terminal_seen_at is None:
                    terminal_seen_at = time.monotonic()
                elif time.monotonic() - terminal_seen_at > 0.75:
                    break

            await asyncio.sleep(0.03)
    except (WebSocketDisconnect, RuntimeError):
        return
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>9-Ball Foul Detector</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #07120f;
      --panel: #0d201a;
      --panel-2: #122a22;
      --line: #214638;
      --text: #eef8f3;
      --muted: #9cb7aa;
      --green: #31d184;
      --yellow: #f4c95d;
      --red: #ff6b6b;
      --blue: #67b7ff;
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, #133b2d 0, transparent 38rem),
        var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    }
    .shell { width: min(1500px, 96vw); margin: 0 auto; padding: 24px 0 40px; }
    header {
      display: flex; align-items: end; justify-content: space-between;
      gap: 18px; margin-bottom: 18px;
    }
    h1 { margin: 0; font-size: clamp(1.55rem, 3vw, 2.35rem); }
    .subtitle { color: var(--muted); margin-top: 7px; }
    .pill {
      border: 1px solid var(--line); border-radius: 999px;
      padding: 8px 13px; background: #091b15; color: var(--muted);
      white-space: nowrap;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 2fr) minmax(320px, 0.8fr);
      gap: 16px;
    }
    .card {
      background: color-mix(in srgb, var(--panel) 94%, transparent);
      border: 1px solid var(--line); border-radius: 16px;
      box-shadow: 0 18px 60px #0005; overflow: hidden;
    }
    .card-body { padding: 16px; }
    .toolbar {
      display: flex; flex-wrap: wrap; align-items: end; gap: 10px;
      padding: 14px 16px; border-bottom: 1px solid var(--line);
      background: var(--panel-2);
    }
    label { display: grid; gap: 6px; color: var(--muted); font-size: .86rem; }
    select, input[type=file], button {
      font: inherit; border-radius: 9px; border: 1px solid var(--line);
      min-height: 40px;
    }
    select, input[type=file] {
      color: var(--text); background: #071712; padding: 8px 10px;
    }
    button {
      color: #062016; background: var(--green); padding: 8px 14px;
      font-weight: 750; cursor: pointer;
    }
    button.secondary { color: var(--text); background: #17392d; }
    button.warning { color: #241800; background: var(--yellow); }
    button.danger { color: white; background: #9e3636; }
    button:disabled { cursor: not-allowed; opacity: .42; }
    .stage {
      display: grid; grid-template-columns: 1fr 1fr; gap: 2px;
      background: #020806; min-height: 430px;
    }
    .viewport { position: relative; display: grid; place-items: center; min-width: 0; }
    .viewport video, .viewport img {
      display: block; width: 100%; height: 100%; max-height: 70vh;
      object-fit: contain; background: #000;
    }
    .viewport-tag {
      position: absolute; left: 10px; top: 10px; padding: 6px 9px;
      border-radius: 7px; background: #000b; color: white; font-size: .76rem;
    }
    .empty {
      color: #708b7f; text-align: center; padding: 30px;
    }
    progress { width: 100%; height: 12px; accent-color: var(--green); }
    .status-line {
      display: flex; justify-content: space-between; gap: 12px;
      margin: 9px 0; color: var(--muted); font-size: .9rem;
    }
    h2 { margin: 0 0 10px; font-size: 1.02rem; }
    .list {
      list-style: none; padding: 0; margin: 0; display: grid; gap: 7px;
      max-height: 260px; overflow: auto;
    }
    .list li {
      padding: 9px 10px; border-radius: 8px; background: #071712;
      border: 1px solid #173529; color: var(--muted); font-size: .86rem;
      overflow-wrap: anywhere;
    }
    .list li.foul { color: #ffdede; border-color: #743737; background: #2b1212; }
    .list li.error { color: #ffdede; border-color: var(--red); }
    .downloads a {
      display: block; color: var(--blue); text-decoration: none;
      padding: 7px 0; overflow-wrap: anywhere;
    }
    .downloads a:hover { text-decoration: underline; }
    .stack { display: grid; gap: 16px; }
    .note { color: var(--muted); font-size: .82rem; line-height: 1.5; }
    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
      .stage { grid-template-columns: 1fr; }
      .viewport { min-height: 280px; }
      header { align-items: start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>9-Ball Foul Detector</h1>
        <div class="subtitle">Windows camera in your browser. Detection inside Docker.</div>
      </div>
      <div class="pill" id="connectionPill">Ready</div>
    </header>

    <div class="grid">
      <section class="card">
        <div class="toolbar">
          <label>Model
            <select id="modelSelect"></select>
          </label>
          <button id="enableCamera">Enable camera</button>
          <button id="startRecording" disabled>Start recording</button>
          <button id="stopRecording" class="danger" disabled>Stop &amp; analyze</button>
          <label>Or analyze a video
            <input id="fileInput" type="file" accept="video/*">
          </label>
          <button id="uploadVideo" class="secondary">Analyze file</button>
        </div>

        <div class="stage">
          <div class="viewport">
            <video id="cameraPreview" autoplay muted playsinline></video>
            <span class="viewport-tag">Camera preview</span>
          </div>
          <div class="viewport">
            <img id="resultFrame" alt="Annotated detection output" hidden>
            <div class="empty" id="resultEmpty">
              Annotated frames will appear here after the recording is uploaded.
            </div>
            <span class="viewport-tag">Detection output</span>
          </div>
        </div>

        <div class="card-body">
          <progress id="progressBar" max="100" value="0"></progress>
          <div class="status-line">
            <span id="statusText">Choose a model and enable the camera.</span>
            <span id="progressText">0%</span>
          </div>
          <div>
            <button id="pauseJob" class="warning" disabled>Pause</button>
            <button id="stopJob" class="danger" disabled>Stop analysis</button>
          </div>
        </div>
      </section>

      <aside class="stack">
        <section class="card card-body">
          <h2>Fouls</h2>
          <ul class="list" id="foulList">
            <li>No fouls reported.</li>
          </ul>
        </section>
        <section class="card card-body">
          <h2>Detection events</h2>
          <ul class="list" id="eventList">
            <li>Waiting for a recording.</li>
          </ul>
        </section>
        <section class="card card-body">
          <h2>Output files</h2>
          <div class="downloads" id="downloads">
            <div class="note">
              The annotated video and logs will be saved to the mounted output folder.
            </div>
          </div>
        </section>
        <section class="card card-body note">
          The browser records first and Docker analyzes the saved clip afterward.
          This preserves the original FPS-based foul logic.
        </section>
      </aside>
    </div>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const terminalStates = new Set(["completed", "stopped", "error"]);
    let cameraStream = null;
    let recorder = null;
    let recordingChunks = [];
    let currentJobId = null;
    let jobSocket = null;
    let latestFrameUrl = null;
    let isPaused = false;
    let recordingStartedAt = 0;

    function setStatus(text, state = "") {
      $("statusText").textContent = text;
      $("connectionPill").textContent = state || "Ready";
    }

    function addListItem(list, text, className = "") {
      if (list.children.length === 1 &&
          (list.firstElementChild.textContent.startsWith("No ") ||
           list.firstElementChild.textContent.startsWith("Waiting"))) {
        list.innerHTML = "";
      }
      const item = document.createElement("li");
      item.textContent = text;
      if (className) item.className = className;
      list.appendChild(item);
      list.scrollTop = list.scrollHeight;
    }

    function resetResults() {
      $("foulList").innerHTML = "<li>No fouls reported.</li>";
      $("eventList").innerHTML = "<li>Waiting for analysis.</li>";
      $("downloads").innerHTML =
        '<div class="note">Outputs will appear when analysis finishes.</div>';
      $("progressBar").value = 0;
      $("progressText").textContent = "0%";
      $("resultFrame").hidden = true;
      $("resultEmpty").hidden = false;
      if (latestFrameUrl) URL.revokeObjectURL(latestFrameUrl);
      latestFrameUrl = null;
    }

    async function loadModels() {
      const response = await fetch("/api/models");
      const payload = await response.json();
      const select = $("modelSelect");
      select.innerHTML = "";
      for (const name of payload.models) {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        option.selected = name === payload.default;
        select.appendChild(option);
      }
      if (!payload.models.length) {
        setStatus("No model weights were found.", "Configuration error");
        for (const id of ["enableCamera", "uploadVideo"]) $(id).disabled = true;
      }
    }

    async function enableCamera() {
      try {
        if (cameraStream) {
          cameraStream.getTracks().forEach((track) => track.stop());
        }
        cameraStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30, max: 30 }
          },
          audio: false
        });
        $("cameraPreview").srcObject = cameraStream;
        $("startRecording").disabled = false;
        setStatus("Camera ready.", "Camera connected");
      } catch (error) {
        setStatus(`Camera error: ${error.message}`, "Camera unavailable");
      }
    }

    function preferredRecordingType() {
      const choices = [
        "video/webm;codecs=vp9",
        "video/webm;codecs=vp8",
        "video/webm"
      ];
      return choices.find((type) => MediaRecorder.isTypeSupported(type)) || "";
    }

    function startRecording() {
      if (!cameraStream) return;
      resetResults();
      recordingChunks = [];
      const mimeType = preferredRecordingType();
      const options = { videoBitsPerSecond: 6000000 };
      if (mimeType) options.mimeType = mimeType;
      recorder = new MediaRecorder(cameraStream, options);
      recorder.addEventListener("dataavailable", (event) => {
        if (event.data && event.data.size > 0) recordingChunks.push(event.data);
      });
      recorder.start(1000);
      recordingStartedAt = performance.now();
      $("startRecording").disabled = true;
      $("stopRecording").disabled = false;
      $("fileInput").disabled = true;
      $("uploadVideo").disabled = true;
      setStatus("Recording in the browser...", "Recording");
    }

    async function stopRecording() {
      if (!recorder || recorder.state === "inactive") return;
      $("stopRecording").disabled = true;
      const stopped = new Promise((resolve) => {
        recorder.addEventListener("stop", resolve, { once: true });
      });
      recorder.stop();
      await stopped;
      const durationSeconds = Math.max(
        0.1,
        (performance.now() - recordingStartedAt) / 1000
      );
      const blob = new Blob(recordingChunks, {
        type: recorder.mimeType || "video/webm"
      });
      const file = new File([blob], "camera-recording.webm", {
        type: blob.type
      });
      const settings = cameraStream.getVideoTracks()[0]?.getSettings() || {};
      const fps = Number(settings.frameRate || 30);
      setStatus(
        `Uploading ${durationSeconds.toFixed(1)} second recording...`,
        "Uploading"
      );
      await submitVideo(file, fps);
      $("startRecording").disabled = false;
      $("fileInput").disabled = false;
      $("uploadVideo").disabled = false;
    }

    async function analyzeSelectedFile() {
      const file = $("fileInput").files[0];
      if (!file) {
        setStatus("Choose a video file first.", "Waiting");
        return;
      }
      resetResults();
      await submitVideo(file, null);
    }

    async function submitVideo(file, fps) {
      const form = new FormData();
      form.append("file", file, file.name);
      form.append("model", $("modelSelect").value);
      if (fps) form.append("fps", String(fps));

      setStatus("Uploading video to Docker...", "Uploading");
      $("pauseJob").disabled = true;
      $("stopJob").disabled = true;

      try {
        const response = await fetch("/api/jobs", { method: "POST", body: form });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Upload failed");
        }
        currentJobId = payload.job_id;
        connectToJob(currentJobId);
      } catch (error) {
        setStatus(`Upload failed: ${error.message}`, "Error");
        addListItem($("eventList"), error.message, "error");
      }
    }

    function connectToJob(jobId) {
      if (jobSocket) jobSocket.close();
      const protocol = location.protocol === "https:" ? "wss" : "ws";
      jobSocket = new WebSocket(`${protocol}://${location.host}/ws/jobs/${jobId}`);
      jobSocket.binaryType = "blob";

      jobSocket.addEventListener("open", () => {
        setStatus("Connected to detector.", "Processing");
        $("stopJob").disabled = false;
      });

      jobSocket.addEventListener("message", (event) => {
        if (typeof event.data === "string") {
          const payload = JSON.parse(event.data);
          if (payload.type === "status") updateJobStatus(payload);
          else if (payload.type === "foul") {
            const prefix = payload.video_time ? `[${payload.video_time}] ` : "";
            addListItem($("foulList"), `${prefix}${payload.message}`, "foul");
          } else {
            addListItem(
              $("eventList"),
              payload.message || JSON.stringify(payload),
              payload.type === "error" ? "error" : ""
            );
          }
          return;
        }

        if (latestFrameUrl) URL.revokeObjectURL(latestFrameUrl);
        latestFrameUrl = URL.createObjectURL(event.data);
        $("resultFrame").src = latestFrameUrl;
        $("resultFrame").hidden = false;
        $("resultEmpty").hidden = true;
      });

      jobSocket.addEventListener("close", () => {
        jobSocket = null;
      });
    }

    function updateJobStatus(payload) {
      const progress = Number(payload.progress || 0);
      $("progressBar").value = progress;
      $("progressText").textContent = `${progress.toFixed(1)}%`;
      setStatus(payload.error || payload.message, payload.status);
      isPaused = Boolean(payload.paused);
      $("pauseJob").textContent = isPaused ? "Resume" : "Pause";
      $("pauseJob").disabled = !["processing", "paused"].includes(payload.status);
      $("stopJob").disabled = terminalStates.has(payload.status);

      if (terminalStates.has(payload.status)) {
        renderDownloads(payload.outputs || []);
        $("fileInput").disabled = false;
        $("uploadVideo").disabled = false;
        $("startRecording").disabled = !cameraStream;
      }
    }

    function renderDownloads(files) {
      const box = $("downloads");
      box.innerHTML = "";
      if (!files.length) {
        box.innerHTML = '<div class="note">No output files were created.</div>';
        return;
      }
      for (const filename of files) {
        const link = document.createElement("a");
        link.href = `/api/jobs/${currentJobId}/files/${encodeURIComponent(filename)}`;
        link.textContent = filename;
        link.download = filename;
        box.appendChild(link);
      }
    }

    async function pauseOrResume() {
      if (!currentJobId) return;
      const action = isPaused ? "resume" : "pause";
      const response = await fetch(`/api/jobs/${currentJobId}/${action}`, {
        method: "POST"
      });
      if (!response.ok) {
        const payload = await response.json();
        setStatus(payload.detail || `${action} failed`, "Error");
      }
    }

    async function stopAnalysis() {
      if (!currentJobId) return;
      await fetch(`/api/jobs/${currentJobId}/stop`, { method: "POST" });
    }

    $("enableCamera").addEventListener("click", enableCamera);
    $("startRecording").addEventListener("click", startRecording);
    $("stopRecording").addEventListener("click", stopRecording);
    $("uploadVideo").addEventListener("click", analyzeSelectedFile);
    $("pauseJob").addEventListener("click", pauseOrResume);
    $("stopJob").addEventListener("click", stopAnalysis);
    window.addEventListener("beforeunload", () => {
      if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
      if (jobSocket) jobSocket.close();
      if (latestFrameUrl) URL.revokeObjectURL(latestFrameUrl);
    });

    loadModels().catch((error) => {
      setStatus(`Startup error: ${error.message}`, "Error");
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        workers=1,
        access_log=True,
    )
