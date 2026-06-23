"""
9-Ball Billiards Foul Detector — Without ByteTrack (cause it dont work)
=============================================================================
"""

import os
import time
import threading
import dataclasses
from datetime import datetime
from typing import Optional, List
from collections import deque, Counter, defaultdict

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np


# ═══════════════════════════ Configuration ════════════════════════════════
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Anchor to this script's location, not the current working directory, so the
# models are found no matter where the app is launched from.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(_SCRIPT_DIR, "..", "Trained_Models")

def _model_pt_path(name):
    """Resolve ../Trained_Models/<name>/weights/<name>.pt, case-insensitively.

    Returns the actual file on disk if a .pt whose stem matches <name> exists
    (ignoring case); otherwise prefers the usual Ultralytics best.pt/last.pt,
    then any .pt; finally the conventional path (for error messages).
    """
    weights_dir = os.path.join(MODELS_ROOT, name, "weights")
    conventional = os.path.join(weights_dir, f"{name}.pt")
    fallback = None
    preferred = None
    try:
        for f in sorted(os.listdir(weights_dir)):
            stem, ext = os.path.splitext(f)
            if ext.lower() != ".pt":
                continue
            if stem.lower() == name.lower():
                return os.path.join(weights_dir, f)   # exact match wins
            if stem.lower() in ("best", "last") and preferred is None:
                preferred = os.path.join(weights_dir, f)  # standard YOLO weights
            if fallback is None:
                fallback = os.path.join(weights_dir, f)  # any .pt as fallback
    except OSError:
        pass
    return preferred or fallback or conventional

DEFAULT_MODEL_NAME = "YOLO11n"
DEFAULT_MODEL_PATH = _model_pt_path(DEFAULT_MODEL_NAME)

CLASS_NAMES = {
    0: "Ball-1", 1: "Ball-2", 2: "Ball-3", 3: "Ball-4", 4: "Ball-5",
    5: "Ball-6", 6: "Ball-7", 7: "Ball-8", 8: "Ball-9",
    9: "cue",   10: "cue_stick", 11: "pocket", 12: "rack",
}
BALL_CLASSES = set(range(9))     # 0–8  numbered object balls
CUE_CLASS    = 9
POCKET_CLASS = 11
RACK_CLASS   = 12

# Indonesian colour label + swatch colour (BGR) for each object-ball class (0-8)
BALL_COLOR_ID = {
    0: "Kuning", 1: "Biru",  2: "Merah",
    3: "Pink",   4: "Ungu",  5: "Hijau",
    6: "Coklat", 7: "Hitam", 8: "Kuning-Hitam",
}
BALL_BGR = {
    0: (0, 255, 255),   # Kuning / yellow
    1: (255, 0, 0),     # Biru / blue
    2: (0, 0, 255),     # Merah / red
    3: (180, 105, 255), # Pink
    4: (211, 0, 148),   # Ungu / purple
    5: (0, 200, 0),     # Hijau / green
    6: (19, 69, 139),   # Coklat / brown
    7: (0, 0, 0),       # Hitam / black
    8: ((0, 255, 255), (0, 0, 0)),  # Kuning-Hitam (yellow over black)
}

# Detection
CONF_THRESHOLD = 0.35

# ── DEBUG_CONTACT overlay (temporary — set False or delete to remove) ──
# Shows the live contact-tracking state (trajectory candidates, balls that have
# moved this shot, potted balls) bottom-left, so we can see which branch a
# missed-hit lands in. Remove this constant + the DEBUG_CONTACT block in _draw().
SHOW_CONTACT_DEBUG = True

# Shot / velocity  (pixels-per-frame, calibrated for ~720p @ 30 fps)
SHOT_START_VEL = 8.0    # cue faster than this → shot has started
SHOT_END_VEL   = 5.0    # cue slower than this counts toward "at rest"

# Timing (all in seconds)
SHOT_END_SECS     = 1.0   # cue at rest this long → shot is over
SCRATCH_REST_SECS = 1.0   # cue resting inside pocket bbox → scratch foul
CUE_MISSING_SECS       = 12.0  # (legacy — superseded by CUE_OFF_TABLE_SECS below)
CUE_OFF_TABLE_SECS     =  1.5  # cue seen, then vanishes AWAY from any pocket this long → off-table foul
POCKET_DISAPPEAR_SECS  =  0.5  # cue seen near pocket then missing → foul
FOUL_BANNER_SECS  = 3.0   # on-screen foul banner duration

# Contact confirmation
MOVEMENT_CONFIRM_FRAMES = 12   # frames to observe ball movement after contact
MOVEMENT_THRESHOLD_PX   =  8   # min pixel displacement to confirm contact
CONTACT_BBOX_PAD        =  4   # extra pixels on cue bbox for overlap test

# Class-vote stability
CLASS_VOTE_WINDOW = 50          # rolling window size (frames) per track ID

# ── Rail-contact foul ─────────────────────────────────────────────────────
RAIL_MARGIN_PX = 40   # tune upward if your camera angle makes rails appear
BOTTOM_RAIL_PAD_PX = 2   # nudge ONLY the bottom rail edge inward (toward table
                          # center) to close a tiny gap so balls resting on the
                          # real bottom cushion register as touching the rail
POST_SHOT_GRACE_FRAMES = 45  # extra frames after cue rests to let balls reach rails (~1.5 s @ 30 fps)

# ── Trajectory-based first-contact detection ──────────────────────────────────
TRAJECTORY_PROXIMITY_PX  = 35   # max px from the cue's projected path for a ball to be a candidate
TRAJECTORY_LOOKAHEAD_PX  = 400  # how far ahead (px) to extend the trajectory line
LINE_OF_SIGHT_REWIND_FRAMES = 4  # frames before first movement used for Newton's-cradle checks
LINE_OF_SIGHT_BLOCK_RADIUS_PX = TRAJECTORY_PROXIMITY_PX
POCKET_REPLACEMENT_OVERLAP = 0.45  # ball overlap ratio needed to treat a missing pocket as potted


# ═══════════════════════════ Geometry helpers ═════════════════════════════

def bbox_center(b):
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)

def pt_dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def boxes_overlap(b1, b2, pad=0):
    """xyxy overlap; optional symmetric pixel padding applied to b1."""
    a = (b1[0] - pad, b1[1] - pad, b1[2] + pad, b1[3] + pad)
    return not (a[2] < b2[0] or b2[2] < a[0] or
                a[3] < b2[1] or b2[3] < a[1])

def bbox_overlap_ratio(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / min(area1, area2)

def bbox_inside(inner, outer):
    """True if the whole `inner` xyxy box lies within the `outer` xyxy box."""
    return (inner[0] >= outer[0] and inner[1] >= outer[1] and
            inner[2] <= outer[2] and inner[3] <= outer[3])

#NEW CHANGES

def pt_to_line_segment_dist(p, v, w):
    """
    Returns the shortest distance from point p to the line segment vw.
    p = object ball center
    v = previous cue center
    w = current cue center
    """
    px, py = p
    vx, vy = v
    wx, wy = w
    
    # Length of the line segment squared
    l2 = (wx - vx)**2 + (wy - vy)**2
    if l2 == 0.0:
        return pt_dist(p, v) # Cue didn't move
        
    # Change (wy - wy) to (wy - vy)
    t = max(0, min(1, ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2))
    
    proj_x = vx + t * (wx - vx)
    proj_y = vy + t * (wy - vy)
    
    return pt_dist(p, (proj_x, proj_y))


# ═══════════════════════════ Data classes ════════════════════════════════

@dataclasses.dataclass
class Track:
    track_id:   int
    bbox:       list    
    stable_cls: int     
    center:     tuple   


@dataclasses.dataclass
class PendingContact:
    contacted_track_id: int
    contacted_class:    int
    pos_at_contact:     tuple          
    created_frame:      int
    shot_target_class:  Optional[int]  


# ═══════════════════════════ Application ═════════════════════════════════
class FoulDetectorApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("9-Ball Foul Detector — Camera Recording")
        self.root.geometry("1280x860")
        self.root.configure(bg="white")

        # Resources set by the user
        self.model:      Optional[YOLO] = None
        self.model_path: str            = DEFAULT_MODEL_PATH
        self.video_path: Optional[str]  = None
        self.output_dir: Optional[str]  = None

        # I/O handles (live while processing)
        self.cap:          Optional[cv2.VideoCapture] = None
        self.video_writer: Optional[cv2.VideoWriter]  = None
        self.foul_log                                 = None  
        self.debug_log                                = None  # NEW: Frame-by-frame debug log                            = None  

        # Playback state
        self.is_playing  = False
        self.is_paused   = False
        self.fps         = 30.0
        self.frame_idx   = 0
        self._foul_count = 0

        # ── Camera / recording state ──
        self.camera_index:   Optional[int] = None
        self.available_cams: list          = []
        self.is_recording:   bool          = False
        self.recorded_path:  Optional[str] = None
        self._record_thread                = None
        self._rec_start_time: float        = 0.0
        # Requested capture: C922 @ 720p60, recorded as H.264 (MPEG-4 AVC)
        self.cap_width  = 1280
        self.cap_height = 720
        self.cap_fps    = 60
        # fps normalisation for detection (camera 60fps -> detect ~30fps)
        self._frame_stride: int = 1
        self._src_frame_no: int = 0

        # Per-track histories
        self.class_history: defaultdict = defaultdict(
            lambda: deque(maxlen=CLASS_VOTE_WINDOW))
        self.ball_positions: defaultdict = defaultdict(
            lambda: deque(maxlen=90))
        self.ball_position_frames: defaultdict = defaultdict(
            lambda: deque(maxlen=90))

        # Shot / foul state
        self.prev_cue_center:        Optional[tuple] = None
        self._last_cue_bbox:         Optional[tuple] = None
        self._last_pocket_bboxes:    list            = []
        self.cue_velocity:           float           = 0.0
        self.shot_in_progress:       bool            = False
        self.lowest_at_shot_start:   Optional[int]   = None
        self.is_break_shot:          bool            = False
        self.rack_frames_since_seen: int             = 999
        self.checked_contact_tids:   set             = set()
        self._low_vel_run:           int             = 0
        self.pending_contacts:       List[PendingContact]   = []
        
        # Rail & Logging requirements
        self.ball_start_info: dict = {}
        self.balls_left_rail_this_shot: set = set()
        self.contact_occurred_this_shot: bool = False
        self.post_contact_rail_hit:      bool = False
        self.post_contact_pocketed:      bool = False
        self.active_ball_ids_at_contact: dict = {}
        
        # Avoid spamming the log for continuous states
        self.event_log: deque            = deque(maxlen=6)
        self.balls_hit_rail_this_shot: set = set()
        self.potted_balls: set           = set()
        self.stable_pockets: list        = []
        self.stable_pocket_bboxes: list  = []
        self.pocket_candidates: dict     = {}

        # Continuous-foul state
        self.cue_missing_frames:   int  = 0
        self.cue_in_pocket_frames: int  = 0
        self._scratch_fired:       bool = False
        self._missing_fired:       bool = False
        self._off_table_fired:     bool = False

        # Pocket-disappear state
        self.cue_was_near_pocket:          bool = False
        self.cue_near_pocket_streak:       int  = 0   # persistence counter for pocket proximity
        self.cue_gone_after_pocket_frames: int  = 0
        self._pocket_disappear_fired:      bool = False
        self._cue_pocket_replace_fired:    bool = False
        self.post_shot_eval_frame:         Optional[int] = None  # deferred rail evaluation
        self._no_legal_ball_hit_fired:     bool = False

        # Table boundary estimate
        self.table_bounds: Optional[tuple] = None

        # Static rail polygon (hexagon through the 6 pocket centers)
        self.rail_polygon: Optional[np.ndarray] = None
        self.rail_locked: bool = False
        self.pocket_positions: dict = {}   # label -> (x, y) center

        # On-screen banners
        self.foul_banner_text:        str = ""
        self.foul_banner_until_frame: int = 0

        self._build_ui()
        self._populate_models()
        self.refresh_cameras()

    # ─────────────────────────────── UI ──────────────────────────────────

    def _build_ui(self):
        bar = tk.Frame(self.root, bg="white", pady=7)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(bar, text="Camera:", bg="white").pack(side=tk.LEFT, padx=(6, 2))
        self.cam_var = tk.StringVar(value="Detecting…")
        self.cam_menu = tk.OptionMenu(bar, self.cam_var, "Detecting…")
        self.cam_menu.config(state=tk.DISABLED)
        self.cam_menu.pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="Refresh", command=self.refresh_cameras
                  ).pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text="Output Folder",
                  command=self.select_output).pack(side=tk.LEFT, padx=5)
        tk.Label(bar, text="Model:", bg="white").pack(side=tk.LEFT, padx=(8, 2))
        self.model_var = tk.StringVar(value="Scanning…")
        self.model_menu = tk.OptionMenu(bar, self.model_var, "Scanning…")
        self.model_menu.config(state=tk.DISABLED)
        self.model_menu.pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="Browse…",
                  command=self.load_model_dialog).pack(side=tk.LEFT, padx=2)

        tk.Frame(bar, bg="#cccccc", width=2, height=36).pack(
            side=tk.LEFT, padx=14, fill=tk.Y)

        self.upload_btn = tk.Button(bar, text="Upload Video",
                                    command=self.upload_video,
                                    state=tk.DISABLED)
        self.upload_btn.pack(side=tk.LEFT, padx=4)
        self.rec_btn = tk.Button(bar, text="Start Recording",
                                 command=self.start_recording,
                                 state=tk.DISABLED)
        self.rec_btn.pack(side=tk.LEFT, padx=4)
        self.stoprec_btn = tk.Button(bar, text="Stop Recording",
                                     command=self.stop_recording,
                                     state=tk.DISABLED)
        self.stoprec_btn.pack(side=tk.LEFT, padx=4)
        self.pause_btn = tk.Button(bar, text="Pause",
                                   command=self.toggle_pause,
                                   state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=4)

        self.status_lbl = tk.Label(bar, text="Loading model…", bg="white")
        self.status_lbl.pack(side=tk.RIGHT, padx=14)

        self.video_lbl = tk.Label(self.root, bg="black")
        self.video_lbl.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        log_frame = tk.Frame(self.root, bg="#12121f", pady=4)
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8)
        tk.Label(log_frame, text="Foul Log", fg="white", bg="#12121f",
                 font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
        self.foul_listbox = tk.Listbox(
            log_frame, height=5,
            bg="#0d0d1f", fg="#ff7070",
            font=("Consolas", 10),
            selectbackground="#2b2b4a",
            highlightthickness=0, borderwidth=0,
        )
        self.foul_listbox.pack(fill=tk.X, pady=2)

    # ─────────────────────── Model / file loading ─────────────────────────

    def _load_model_async(self, path: str):
        def _do():
            try:
                # Drop any previously loaded model before loading the new one.
                self.model = None

                m = YOLO(path)   # Ultralytics .pt model (auto-selects GPU/CPU)
                self.model      = m
                self.model_path = path
                self.root.after(0, lambda: self.status_lbl.config(
                    text=f"{os.path.basename(path)}", fg="black"))
                self.root.after(0, self._check_ready)
            except Exception as exc:
                msg = str(exc)
                self.root.after(0, lambda: self.status_lbl.config(
                    text=f"Model error: {msg}", fg="#ff7070"))
        threading.Thread(target=_do, daemon=True).start()

    def load_model_dialog(self):
        if self.is_playing:
            messagebox.showwarning(
                "Busy",
                "Stop the current job before loading a different model.")
            return
        path = filedialog.askopenfilename(
            title="Select YOLO weights (.pt)",
            filetypes=[("Ultralytics YOLO weights", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.model_var.set(os.path.basename(path))
            self.status_lbl.config(text="Loading…", fg="black")
            self._load_model_async(path)

    def _discover_models(self):
        """Find models stored as ../Trained_Models/<name>/weights/<name>.pt."""
        names = []
        try:
            for entry in sorted(os.listdir(MODELS_ROOT)):
                if os.path.isfile(_model_pt_path(entry)):
                    names.append(entry)
        except OSError:
            pass
        return names

    def _populate_models(self):
        """Fill the model dropdown from disk and load the default selection."""
        names = self._discover_models()
        menu = self.model_menu["menu"]
        menu.delete(0, tk.END)

        if not names:
            self.model_var.set("No models found")
            self.model_menu.config(state=tk.DISABLED)
            self.status_lbl.config(
                text=f"No models in {MODELS_ROOT} — use Browse…", fg="#ff7070")
            return

        for name in names:
            menu.add_command(
                label=name,
                command=lambda n=name: self._select_model(n))
        self.model_menu.config(state=tk.NORMAL)

        # Start with no model equipped — the user must pick one explicitly.
        self.model_var.set("Select a model")
        self.status_lbl.config(text="Select a model to begin", fg="black")

    def _select_model(self, name):
        if self.is_playing:
            messagebox.showwarning(
                "Busy",
                "Stop the current job before loading a different model.")
            return
        self.model_var.set(name)
        self.status_lbl.config(text="Loading…", fg="black")
        self._load_model_async(_model_pt_path(name))

    def refresh_cameras(self):
        """Probe video devices in a background thread and fill the dropdown."""
        self.status_lbl.config(text="Scanning for cameras…", fg="black")
        threading.Thread(target=self._detect_cameras_worker, daemon=True).start()

    def _detect_cameras_worker(self):
        found = []
        for idx in range(6):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(idx)
            if cap is not None:
                cap.release()
        self.root.after(0, lambda: self._populate_cameras(found))

    def _populate_cameras(self, found):
        self.available_cams = found
        menu = self.cam_menu["menu"]
        menu.delete(0, tk.END)
        if not found:
            self.cam_var.set("No camera found")
            self.cam_menu.config(state=tk.DISABLED)
            self.camera_index = None
            self.status_lbl.config(
                text="No camera detected — check the connection", fg="#ff7070")
            self._check_ready()
            return
        for idx in found:
            label = f"Camera {idx}"
            menu.add_command(
                label=label,
                command=lambda i=idx, l=label: self._select_camera(i, l))
        self._select_camera(found[0], f"Camera {found[0]}")
        self.cam_menu.config(state=tk.NORMAL)
        self.status_lbl.config(text=f"{len(found)} camera(s) found", fg="black")

    def _select_camera(self, idx, label):
        self.camera_index = idx
        self.cam_var.set(label)
        self._check_ready()

    def select_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir = path
            self._check_ready()

    def _check_ready(self):
        ready = (self.camera_index is not None
                 and self.output_dir and self.model is not None
                 and not self.is_recording)
        if hasattr(self, "rec_btn"):
            self.rec_btn.config(state=tk.NORMAL if ready else tk.DISABLED)
        # Uploading a file needs only an output folder + model (no camera).
        upload_ready = bool(self.output_dir and self.model is not None
                            and not self.is_recording and not self.is_playing)
        if hasattr(self, "upload_btn"):
            self.upload_btn.config(
                state=tk.NORMAL if upload_ready else tk.DISABLED)

    # ─────────────────────── Processing lifecycle ─────────────────────────

    # ── Upload an existing video ──────────────────────────────────────────
    def upload_video(self):
        """Pick a video file from disk and run detection on it, reusing the
        same pipeline as a fresh recording (no camera required)."""
        if self.is_recording or self.is_playing:
            messagebox.showwarning(
                "Busy", "Stop the current job before uploading a video.")
            return
        if not (self.output_dir and self.model):
            messagebox.showerror(
                "Not ready",
                "Select an output folder and load a model first.")
            return
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[
                ("Video files",
                 "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.flv *.mpg *.mpeg"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        # Reuse the record->detect path. _begin_detection opens the file, sets
        # up the annotated writer + logs, and starts the processing thread.
        self.recorded_path = path
        self.cam_menu.config(state=tk.DISABLED)
        self._begin_detection(path)

    # ── Recording ────────────────────────────────────────────────────────
    def start_recording(self):
        if self.is_recording:
            return
        if not (self.camera_index is not None and self.output_dir and self.model):
            messagebox.showerror(
                "Not ready",
                "Select a camera, an output folder, and load a model first.")
            return

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror(
                "Error", f"Could not open camera {self.camera_index}.")
            return
        # MJPG is required for the C922 to deliver 720p at 60fps over USB; the
        # default YUYV format caps out around 10fps at this resolution.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cap_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_height)
        cap.set(cv2.CAP_PROP_FPS,          self.cap_fps)

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or self.cap_width
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.cap_height
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1:
            fps = float(self.cap_fps)
        self.cap = cap
        self.fps = fps

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recorded_path = os.path.join(self.output_dir, f"recording_{ts}.mp4")
        writer, codec = self._make_h264_writer(self.recorded_path, fps, (W, H))
        if writer is None:
            cap.release()
            self.cap = None
            messagebox.showerror("Error", "Could not open a video writer.")
            return
        self.video_writer = writer

        self.is_recording    = True
        self._rec_start_time = time.time()
        self._rec_frame_count = 0
        self.rec_btn.config(state=tk.DISABLED)
        self.stoprec_btn.config(state=tk.NORMAL)
        self.cam_menu.config(state=tk.DISABLED)
        self.status_lbl.config(
            text=f"REC  {W}x{H} @ {fps:.0f}fps  ({codec})", fg="#ff7070")

        self._record_thread = threading.Thread(
            target=self._record_loop, daemon=True)
        self._record_thread.start()

    def _record_loop(self):
        try:
            while self.is_recording and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                if self.video_writer:
                    self.video_writer.write(frame)
                    self._rec_frame_count += 1
                # Live preview with a REC dot + elapsed time.
                prev = frame.copy()
                elapsed = time.time() - self._rec_start_time
                cv2.circle(prev, (32, 32), 12, (0, 0, 255), -1)
                cv2.putText(
                    prev, f"REC  {int(elapsed // 60):02d}:{int(elapsed % 60):02d}",
                    (54, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                self._display_frame(prev)
        finally:
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self._rec_stop_time = time.time()
        self.stoprec_btn.config(state=tk.DISABLED)
        self.status_lbl.config(text="Finishing recording…", fg="black")
        # Close the file off the UI thread, then run detection on it.
        threading.Thread(target=self._finish_and_detect, daemon=True).start()

    def _finish_and_detect(self):
        rt = self._record_thread
        if rt is not None and rt.is_alive():
            rt.join(timeout=5.0)
        path = self.recorded_path
        if not path or not os.path.exists(path):
            self.root.after(0, lambda: self.status_lbl.config(
                text="Recording file missing — nothing to detect", fg="#ff7070"))
            self.root.after(0, self._check_ready)
            return

        # The camera's nominal fps (CAP_PROP_FPS) is usually wrong — a USB
        # webcam rarely delivers its requested rate, so the recording was tagged
        # too fast and plays at "super speed". Recover the rate we ACTUALLY
        # captured at (frames written / wall-clock seconds) and use that for the
        # annotated output so it plays back at real time.
        elapsed = max(1e-6, getattr(self, "_rec_stop_time", time.time())
                      - self._rec_start_time)
        frames  = getattr(self, "_rec_frame_count", 0)
        real_fps = frames / elapsed if frames > 0 else None
        if real_fps is not None and not (1.0 <= real_fps <= 1000.0):
            real_fps = None
        self.root.after(0, lambda: self._begin_detection(path, fps_override=real_fps))

    def _make_h264_writer(self, path, fps, size):
        """Open an H.264 (MPEG-4 AVC) writer, falling back gracefully if the
        OpenCV build has no AVC encoder."""
        for codec in ("avc1", "H264", "X264"):
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
            if w.isOpened():
                return w, codec
            w.release()
        # Fallback: MPEG-4 Part 2 (not AVC, but always available).
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if w.isOpened():
            return w, "mp4v (H264 unavailable)"
        return None, None

    # ── Detection on the just-recorded file ───────────────────────────────
    def _begin_detection(self, video_path, fps_override=None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open the recorded video.")
            self._check_ready()
            return

        # Prefer the measured capture rate (from recording) over the file's
        # tagged fps, which is unreliable for webcam recordings. Uploaded files
        # pass no override, so their real fps tag is trusted.
        if fps_override and 1.0 <= fps_override <= 1000.0:
            src_fps = fps_override
        else:
            src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not src_fps or src_fps < 1 or src_fps > 1000:
            src_fps = 30.0
        W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # The detector's velocity thresholds are px-per-frame, calibrated for
        # ~30fps. The recording is 60fps, so process every Nth frame to bring
        # the effective rate back to ~30fps and keep that math valid.
        self._frame_stride = max(1, int(round(src_fps / 30.0)))
        self._src_frame_no = 0
        self._src_fps      = src_fps              # true source rate (for output)
        self._last_annotated = None
        self.fps = src_fps / self._frame_stride   # effective (processed) fps

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer, _ = self._make_h264_writer(
            os.path.join(self.output_dir, f"annotated_{ts}.mp4"),
            self._src_fps, (W, H))
        self.video_writer = writer
        self.foul_log = open(
            os.path.join(self.output_dir, f"foul_log_{ts}.txt"),
            "w", encoding="utf-8",
        )
        self.foul_log.write(
            f"Foul Log — {ts}\nSource: {self.video_path}\n\n")

        self.debug_log = open(
            os.path.join(self.output_dir, f"debug_positions_{ts}.txt"),
            "w", encoding="utf-8",
        )
        self.debug_log.write(
            f"Ball Position Debug Log — {ts}\nSource: {self.video_path}\n\n")

        self._reset_state()

        self.is_playing = True
        self.is_paused  = False
        self.rec_btn.config(state=tk.DISABLED)
        self.stoprec_btn.config(state=tk.DISABLED)
        self.upload_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL, text="Pause")
        self.status_lbl.config(
            text=f"Detecting on recording (stride {self._frame_stride})…",
            fg="black")

        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()

    def _reset_state(self):
        self.frame_idx               = 0
        self._foul_count             = 0
        self.class_history           = defaultdict(
            lambda: deque(maxlen=CLASS_VOTE_WINDOW))
        self.ball_positions          = defaultdict(lambda: deque(maxlen=90))
        self.ball_position_frames    = defaultdict(lambda: deque(maxlen=90))
        self.prev_cue_center         = None
        self._last_cue_bbox          = None
        self._last_pocket_bboxes     = []
        self.cue_velocity            = 0.0
        self.shot_in_progress        = False
        self.lowest_at_shot_start    = None
        self.is_break_shot           = False
        self.rack_frames_since_seen  = 999
        self.checked_contact_tids    = set()
        self.balls_near_pocket           = {}
        self.disappeared_near_pocket_frames = defaultdict(int)
        self._low_vel_run            = 0
        self.pending_contacts        = []
        self.stable_pockets: list = []       
        self.stable_pocket_bboxes: list = []
        self.pocket_candidates: dict = {}    
        self.balls_moved_this_shot: dict = {}
        self.trajectory_candidates: set  = set()   # ball IDs in/near cue path this shot
        self.first_trajectory_hit_fired: bool = False
        self._no_legal_ball_hit_fired = False
        
        # Rail requirements & logs variables
        self.ball_start_info = {}
        self.balls_left_rail_this_shot.clear()
        self.contact_occurred_this_shot  = False
        self.post_contact_rail_hit       = False
        self.post_contact_pocketed       = False
        self.active_ball_ids_at_contact  = {}
        self.event_log.clear()
        self.balls_hit_rail_this_shot.clear()
        self.potted_balls.clear()

        self.cue_missing_frames      = 0
        self.cue_in_pocket_frames    = 0
        self._scratch_fired          = False
        self._missing_fired          = False
        self._off_table_fired        = False
        self.cue_was_near_pocket          = False
        self.cue_near_pocket_streak       = 0
        self.cue_gone_after_pocket_frames = 0
        self._pocket_disappear_fired      = False
        self._cue_pocket_replace_fired    = False
        self.post_shot_eval_frame         = None
        self._last_annotated              = None
        self.table_bounds                 = None   
        self.rail_polygon                 = None
        self.rail_locked                  = False
        self.pocket_positions             = {}
        self.foul_banner_text             = ""
        self.foul_banner_until_frame      = 0
        self.foul_listbox.delete(0, tk.END)

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.pause_btn.config(
            text="Resume" if self.is_paused else "Pause")

    def stop_processing(self):
        self.is_playing = False

    def _cleanup(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        if self.foul_log:
            self.foul_log.close()
            self.foul_log = None
        if self.debug_log:
            self.debug_log.close()
            self.debug_log = None
        # The loaded YOLO model stays resident between videos so the next job
        # can start immediately; nothing hardware-related to release here.

        ready = bool(self.camera_index is not None
                     and self.output_dir and self.model)
        self.rec_btn.config(state=tk.NORMAL if ready else tk.DISABLED)
        self.stoprec_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED, text="Pause")
        if hasattr(self, "upload_btn"):
            self.upload_btn.config(
                state=tk.NORMAL if (self.output_dir and self.model)
                else tk.DISABLED)
        if hasattr(self, "cam_menu") and self.available_cams:
            self.cam_menu.config(state=tk.NORMAL)

    # ─────────────────────── Rail & Log helpers ───────────────────────────

    def _update_table_bounds(self, pockets: list):
        if len(self.stable_pocket_bboxes) != 6 and len(pockets) == 6:
            ordered = sorted(pockets, key=lambda p: (p.center[1], p.center[0]))
            self.stable_pocket_bboxes = [list(p.bbox) for p in ordered]
            self.stable_pockets = [bbox_center(b) for b in self.stable_pocket_bboxes]
            self._add_event_log("6 pocket locations locked")

        if not pockets:
            return
            
        # ── 1. Register new pockets and update existing ones ──
        for p in pockets:
            px, py = p.center
            matched = False
            
            for i, (sx, sy) in enumerate(self.stable_pockets):
                if pt_dist((px, py), (sx, sy)) < 60:  
                    self.stable_pockets[i] = (sx * 0.95 + px * 0.05, sy * 0.95 + py * 0.05)
                    matched = True
                    break
                    
            if not matched:
                matched_candidate = None
                for (cx, cy) in list(self.pocket_candidates.keys()):
                    if pt_dist((px, py), (cx, cy)) < 60:
                        self.pocket_candidates[(cx, cy)] += 1
                        matched_candidate = (cx, cy)
                        if self.pocket_candidates[(cx, cy)] > 15:
                            self.stable_pockets.append((cx, cy))
                            del self.pocket_candidates[(cx, cy)]
                        break
                
                if not matched_candidate:
                    self.pocket_candidates[(px, py)] = 1

        # ── 2. Form the table bounds EXCLUSIVELY from stable pockets ──
        if len(self.stable_pockets) >= 4:
            xs = [sx for (sx, sy) in self.stable_pockets]
            ys = [sy for (sx, sy) in self.stable_pockets]
            
            self.table_bounds = (min(xs), min(ys), max(xs), max(ys))

        # ── 3. Build the static rail polygon ONCE we have all 6 pockets ──
        if not self.rail_locked and len(self.stable_pockets) == 6:
            self._build_static_rail(self.stable_pockets)

    def _build_static_rail(self, centers: list):
        """Classify the 6 pocket centers and freeze a static hexagonal rail.

        The table is landscape and roughly axis-aligned, so the two middle
        pockets sit on the top and bottom long rails. Sort top-to-bottom,
        then left-to-right within each row.
        """
        if len(centers) != 6:
            return

        by_y   = sorted(centers, key=lambda c: c[1])
        top    = sorted(by_y[:3], key=lambda c: c[0])    # left -> right
        bottom = sorted(by_y[3:], key=lambda c: c[0])    # left -> right

        self.pocket_positions = {
            "top_left":      top[0],
            "top_middle":    top[1],
            "top_right":     top[2],
            "bottom_left":   bottom[0],
            "bottom_middle": bottom[1],
            "bottom_right":  bottom[2],
        }

        # Walk the perimeter. Bottom-middle is excluded: it sits inboard of
        # the bottom rail line and would notch the polygon, so the bottom edge
        # runs straight from bottom-right to bottom-left.
        # Nudge only the bottom rail edge inward (smaller y == toward table
        # center) so balls on the real bottom cushion close the predicted gap.
        br = (bottom[2][0], bottom[2][1] - BOTTOM_RAIL_PAD_PX)
        bl = (bottom[0][0], bottom[0][1] - BOTTOM_RAIL_PAD_PX)
        ordered = [
            top[0], top[1], top[2],   # TL -> TM -> TR
            br, bl,                   # BR -> BL (bottom rail padded in)
        ]
        self.rail_polygon = np.array(
            [(int(round(x)), int(round(y))) for (x, y) in ordered],
            dtype=np.int32,
        )
        self.rail_locked = True
        self._add_event_log("Static rail locked (6 pockets)")

    def _dist_to_rail(self, point) -> Optional[float]:
        """Shortest distance from a point to any edge of the static rail."""
        if self.rail_polygon is None:
            return None
        pts = self.rail_polygon
        n   = len(pts)
        return min(
            pt_to_line_segment_dist(point, tuple(pts[i]), tuple(pts[(i + 1) % n]))
            for i in range(n)
        )

    def _ball_at_rail(self, bbox: list, frame_shape: tuple) -> bool:
        bx1, by1, bx2, by2 = bbox

        # Preferred: distance from the ball to the static rail polygon edges.
        if self.rail_polygon is not None:
            cx, cy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
            radius = ((bx2 - bx1) + (by2 - by1)) * 0.25
            d = self._dist_to_rail((cx, cy))
            return d is not None and d <= RAIL_MARGIN_PX + radius

        # Fallback (rail not locked yet): old axis-aligned rectangle test.
        h, w = frame_shape[:2]
        if self.table_bounds is not None:
            rx1, ry1, rx2, ry2 = self.table_bounds
        else:
            rx1, ry1, rx2, ry2 = 0, 0, w, h

        return (
            bx1 <= rx1 + RAIL_MARGIN_PX or
            bx2 >= rx2 - RAIL_MARGIN_PX or
            by1 <= ry1 + RAIL_MARGIN_PX or
            by2 >= ry2 - RAIL_MARGIN_PX
        )
        
    def _add_event_log(self, msg: str):
        """Pushes a timed event string to the top-left visual log."""
        t      = self.frame_idx / max(self.fps, 1e-6)
        ts_str = f"[{int(t // 60):02d}:{int(t % 60):02d}]"
        self.event_log.append(f"{ts_str} {msg}")

    def _mark_ball_potted(self, tid: int, cls: int):
        if tid in self.potted_balls:
            return
        self.potted_balls.add(tid)
        if self.contact_occurred_this_shot:
            self.post_contact_pocketed = True
        name = CLASS_NAMES.get(cls, "ball")
        self._add_event_log(f"{name} potted")

    def _ball_replaced_saved_pocket(self, ball: Track, pockets: List[Track]) -> bool:
        if len(self.stable_pocket_bboxes) != 6:
            return False

        for pocket_bbox in self.stable_pocket_bboxes:
            pocket_visible = any(
                bbox_overlap_ratio(p.bbox, pocket_bbox) >= 0.35
                for p in pockets
            )
            if pocket_visible:
                continue
            if bbox_overlap_ratio(ball.bbox, pocket_bbox) >= POCKET_REPLACEMENT_OVERLAP:
                return True
        return False

    def _evaluate_rail_checks(self):
        if not self.contact_occurred_this_shot:
            return

        # If any ball was seen overlapping a pocket, it is potted -> exempt
        if self.post_contact_pocketed:
            return

        # If a rail was hit by any ball -> exempt
        if self.post_contact_rail_hit:
            return

        # Otherwise -> FOUL
        self._register_foul("No rail contact: Neither the cue nor object balls reached a rail or pocket after contact")

    def _register_no_legal_ball_hit(self):
        if self._no_legal_ball_hit_fired:
            return
        self._register_foul("No legal ball hit")
        self._no_legal_ball_hit_fired = True

    # ─────────────────────── Main processing loop ─────────────────────────

    def _history_center_at_frame(self, tid: int, target_frame: int, fallback=None):
        hist = self.ball_positions.get(tid)
        if not hist:
            return fallback

        frames = self.ball_position_frames.get(tid)
        if frames and len(frames) == len(hist):
            for idx in range(len(frames) - 1, -1, -1):
                if frames[idx] <= target_frame:
                    return hist[idx]
            return fallback if fallback is not None else hist[0]

        frames_back = self.frame_idx - target_frame
        if frames_back < 0:
            return hist[-1]
        idx = len(hist) - 1 - frames_back
        if 0 <= idx < len(hist):
            return hist[idx]
        return fallback if fallback is not None else hist[0]

    def _line_of_sight_first_hit(self, moved_tid: int):
        moved_frame = self.balls_moved_this_shot.get(moved_tid)
        moved_info = self.ball_start_info.get(moved_tid)
        if moved_frame is None or moved_info is None:
            return moved_tid, None

        cue_tid = None
        for tid, info in self.ball_start_info.items():
            if info.get("cls") == CUE_CLASS:
                cue_tid = tid
                break
        if cue_tid is None:
            return moved_tid, None

        cue_info = self.ball_start_info.get(cue_tid, {})
        snapshot_frame = moved_frame - LINE_OF_SIGHT_REWIND_FRAMES
        cue_pos = self._history_center_at_frame(
            cue_tid, snapshot_frame, cue_info.get("center"))
        moved_pos = self._history_center_at_frame(
            moved_tid, snapshot_frame, moved_info.get("center"))
        if cue_pos is None or moved_pos is None:
            return moved_tid, None

        vx = moved_pos[0] - cue_pos[0]
        vy = moved_pos[1] - cue_pos[1]
        line_len2 = vx * vx + vy * vy
        if line_len2 < 1.0:
            return moved_tid, None

        blockers = []
        for tid, info in self.ball_start_info.items():
            if tid in (cue_tid, moved_tid):
                continue
            if info.get("cls") not in BALL_CLASSES:
                continue
            center = self._history_center_at_frame(
                tid, snapshot_frame, info.get("center"))
            if center is None:
                continue

            wx = center[0] - cue_pos[0]
            wy = center[1] - cue_pos[1]
            t = (wx * vx + wy * vy) / line_len2
            if t <= 0.0 or t >= 1.0:
                continue

            proj = (cue_pos[0] + t * vx, cue_pos[1] + t * vy)
            dist_to_line = pt_dist(center, proj)
            if dist_to_line <= LINE_OF_SIGHT_BLOCK_RADIUS_PX:
                blockers.append((pt_dist(cue_pos, center), tid))

        if not blockers:
            return moved_tid, None

        blockers.sort()
        blocker_tid = blockers[0][1]
        return blocker_tid, moved_tid

    def _process_loop(self):
        scratch_frames          = max(1, int(SCRATCH_REST_SECS    * self.fps))
        off_table_thresh        = max(1, int(CUE_OFF_TABLE_SECS   * self.fps))
        shot_end_frames         = max(1, int(SHOT_END_SECS        * self.fps))
        pocket_disappear_thresh = max(1, int(POCKET_DISAPPEAR_SECS * self.fps))

        try:
            while self.is_playing and self.cap and self.cap.isOpened():
                if self.is_paused:
                    time.sleep(0.04)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    break
                # Frame-stride: skip frames so the effective detection rate is
                # ~30fps even though the recording is 60fps (keeps the
                # px-per-frame velocity thresholds calibrated correctly).
                self._src_frame_no += 1
                if (self._frame_stride > 1
                        and (self._src_frame_no % self._frame_stride) != 0):
                    # Skipped for detection, but still emit a frame (reusing the
                    # last annotated image) so the output keeps the same
                    # duration as the source video.
                    if self.video_writer is not None:
                        self.video_writer.write(
                            self._last_annotated
                            if self._last_annotated is not None else frame)
                    continue
                self.frame_idx += 1

                # ── 1. Detect + Track ──
                try:
                    yolo_out = self.model.predict(
                        frame, verbose=False,
                        conf=CONF_THRESHOLD,
                    )[0]
                except Exception as exc:
                    err_msg = f"Inference Error: {exc}"
                    print(err_msg)
                    
                    # Force the frame to display with the error text overlaid
                    cv2.putText(frame, err_msg[:70], (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    self._display_frame(frame)
                    
                    # Briefly pause so the loop doesn't spin at lightning speed
                    time.sleep(0.03) 
                    continue

                # ── 2. Parse tracks ───────────────────────────────────────
                balls:      List[Track] = []
                cue:        Optional[Track] = None
                pockets:    List[Track] = []
                cue_sticks: List[Track] = []
                
                rack_detected_this_frame = False

                if len(yolo_out.boxes) > 0:
                    # Sort boxes by confidence so we process the most confident detections first
                    sorted_boxes = sorted(yolo_out.boxes, key=lambda b: b.conf.item(), reverse=True)
                    seen_ball_classes = set()

                    for i, box in enumerate(sorted_boxes):
                        raw_cls = int(box.cls.item())
                        
                        # --- THE FIX: Use the unique ball class as the permanent ID ---
                        if raw_cls in BALL_CLASSES or raw_cls == CUE_CLASS:
                            if raw_cls in seen_ball_classes:
                                continue # Skip lower-confidence duplicate detections (ghost balls)
                            seen_ball_classes.add(raw_cls)
                            tid = raw_cls # e.g., the 9-ball will ALWAYS be ID 8.
                        else:
                            tid = 1000 + i # Give pockets and sticks arbitrary IDs so they don't clash
                        # --------------------------------------------------------------

                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                        raw_cls = int(box.cls.item())
                        bbox    = [x1, y1, x2, y2]
                        center  = bbox_center(bbox)

                        stable_cls = raw_cls
                        self.ball_positions[tid].append(center)
                        self.ball_position_frames[tid].append(self.frame_idx)

                        t = Track(track_id=tid, bbox=bbox,
                                  stable_cls=stable_cls, center=center)

                        if stable_cls in BALL_CLASSES:
                            balls.append(t)
                        elif stable_cls == CUE_CLASS:
                            if cue is None:
                                cue = t
                        elif stable_cls == POCKET_CLASS:
                            pockets.append(t)
                        elif stable_cls == 10:
                            cue_sticks.append(t)
                        elif stable_cls == RACK_CLASS:
                            rack_detected_this_frame = True
                            
                # Update rack visibility history (30-frame rolling window protects against brief detection flickers)
                if rack_detected_this_frame:
                    self.rack_frames_since_seen = 0
                else:
                    self.rack_frames_since_seen += 1

                lowest_ball = min((b.stable_cls for b in balls), default=None)
                ball_by_id  = {b.track_id: b for b in balls}

                # --- DEBUG LOGGING ---
                if self.debug_log:
                    self.debug_log.write(f"--- Frame {self.frame_idx} ---\n")
                    if self.table_bounds:
                        rx1, ry1, rx2, ry2 = self.table_bounds
                        self.debug_log.write(f"  Table Bounds: ({rx1:.1f}, {ry1:.1f}) to ({rx2:.1f}, {ry2:.1f})\n")
                    else:
                        self.debug_log.write("  Table Bounds: None\n")
                        
                    for b in balls:
                        # FILTER: Only log the 7-ball (class 6) and 9-ball (class 8)
                        if b.stable_cls in (6, 8):
                            ball_name = "7-ball" if b.stable_cls == 6 else "9-ball"
                            self.debug_log.write(f"  {ball_name:<6} (ID: {b.track_id:3d}) | Center: ({b.center[0]:6.1f}, {b.center[1]:6.1f}) | BBox: {b.bbox}\n")
                    
                    if cue:
                        self.debug_log.write(f"  CUE    (ID: {cue.track_id:3d}) | Center: ({cue.center[0]:6.1f}, {cue.center[1]:6.1f}) | BBox: {cue.bbox}\n")
                    self.debug_log.write("\n")
                # ---------------------

                # ── 3a. Update table boundary estimate from pockets ───────

                # ── 3a. Update table boundary estimate from pockets ───────
                self._update_table_bounds(pockets)
                
                # ── 3b. Global Pocketing Check (every frame) ──────────────
                current_ball_ids = {b.track_id for b in balls}

                for b in balls:
                    if b.track_id not in self.potted_balls:
                        if self._ball_replaced_saved_pocket(b, pockets):
                            self._mark_ball_potted(b.track_id, b.stable_cls)
                            self.balls_near_pocket.pop(b.track_id, None)
                            self.disappeared_near_pocket_frames.pop(b.track_id, None)
                            continue

                        if any(boxes_overlap(b.bbox, p.bbox, pad=10) for p in pockets):
                            self.balls_near_pocket[b.track_id] = b.stable_cls
                        else:
                            self.balls_near_pocket.pop(b.track_id, None)
                            self.disappeared_near_pocket_frames.pop(b.track_id, None)

                # Cue-ball pocket replacement: a pocket box vanished and the cue
                # ball now occupies its last known location -> the cue was potted,
                # which is a scratch foul.  Mirrors the object-ball logic above
                # (that loop can't catch the cue, since it is never in `balls`).
                if (cue is not None
                        and not self._cue_pocket_replace_fired
                        and self._ball_replaced_saved_pocket(cue, pockets)):
                    self._register_foul("Scratch: cue ball entered pocket")
                    self._cue_pocket_replace_fired = True
                    self._add_event_log("Cue ball scratch")

                for tid, cls in list(self.balls_near_pocket.items()):
                    if tid not in current_ball_ids:
                        self.disappeared_near_pocket_frames[tid] += 1
                        
                        if self.disappeared_near_pocket_frames[tid] > 5:
                            self._mark_ball_potted(tid, cls)
                            self.balls_near_pocket.pop(tid, None)
                            self.disappeared_near_pocket_frames.pop(tid, None)
                    else:
                        self.disappeared_near_pocket_frames[tid] = 0

                if self.potted_balls:
                    balls = [b for b in balls if b.track_id not in self.potted_balls]
                    ball_by_id = {b.track_id: b for b in balls}
                    lowest_ball = min((b.stable_cls for b in balls), default=None)

                # ── 3c. Global Rail Check ─────────────────────────────────
                all_moving = balls + ([cue] if cue else [])
                
                # 1. Track if balls that started on the rail have explicitly left it
                for b in all_moving:
                    start_info = self.ball_start_info.get(b.track_id)
                    if start_info and start_info["on_rail"]:
                        if b.track_id not in self.balls_left_rail_this_shot:
                            # To leave the rail, it must clearly exit the padded boundary OR move > 40px
                            left_zone = False
                            if self.rail_polygon is not None:
                                bx1, by1, bx2, by2 = b.bbox
                                radius = ((bx2 - bx1) + (by2 - by1)) * 0.25
                                d = self._dist_to_rail(b.center)
                                left_zone = (d is not None and
                                             d > RAIL_MARGIN_PX + 10 + radius)
                            elif self.table_bounds is not None:
                                rx1, ry1, rx2, ry2 = self.table_bounds
                                bx1, by1, bx2, by2 = b.bbox
                                m = RAIL_MARGIN_PX + 10
                                left_zone = (bx1 > rx1 + m and bx2 < rx2 - m and
                                             by1 > ry1 + m and by2 < ry2 - m)
                                             
                            dist_moved = pt_dist(b.center, start_info["center"])
                            if left_zone or dist_moved > 40:
                                self.balls_left_rail_this_shot.add(b.track_id)

                # 2. Check for new rail hits during the shot (and its post-shot
                #    grace window), even if the cue never contacted a ball -- a
                #    cue that only rolls into a rail must still be detected.
                if (self.shot_in_progress
                        or self.post_shot_eval_frame is not None):
                    for b in all_moving:
                        if b.track_id not in self.balls_hit_rail_this_shot:
                            if self._ball_at_rail(b.bbox, frame.shape):
                                
                                start_center = None
                                started_on_rail = False
                                
                                # Resolve physical start location (handles ID tracking switches)
                                if b.track_id in self.ball_start_info:
                                    start_center = self.ball_start_info[b.track_id]["center"]
                                    started_on_rail = self.ball_start_info[b.track_id]["on_rail"]
                                elif self.ball_start_info:
                                    # ID changed mid-shot -> find nearest starting ball
                                    _, nearest_info = min(
                                        self.ball_start_info.items(),
                                        key=lambda item: pt_dist(b.center, item[1]["center"])
                                    )
                                    # If within 40px, we assume it's the same physical ball
                                    if pt_dist(b.center, nearest_info["center"]) < 40:
                                        start_center = nearest_info["center"]
                                        started_on_rail = nearest_info["on_rail"]

                                valid_rail_hit = False
                                
                                if start_center is None:
                                    # 🚨 FIX: The ball was likely missed by YOLO on the exact frame the 
                                    # shot started. Instead of assuming it flew into the rail, we heal 
                                    # the snapshot by registering its current position as its start location.
                                    self.ball_start_info[b.track_id] = {
                                        "center": b.center,
                                        "on_rail": True
                                    }
                                else:
                                    dist_moved = pt_dist(b.center, start_center)
                                    
                                    # MUST have physically moved >= 15 pixels.
                                    if dist_moved >= 60:
                                        if started_on_rail:
                                            # If it started on the rail, require strict proof it left and returned
                                            if b.track_id in self.balls_left_rail_this_shot or dist_moved > 60:
                                                valid_rail_hit = True
                                        else:
                                            # Started OFF rail, moved >15px, and is now ON rail
                                            valid_rail_hit = True

                                if valid_rail_hit:
                                    self.balls_hit_rail_this_shot.add(b.track_id)
                                    self.post_contact_rail_hit = True
                                    name = CLASS_NAMES.get(b.stable_cls, "ball")
                                    self._add_event_log(f"{name} hit rail")

                # ── 4. Foul logic ─────────────────────────────────────────

                # Remember the final-frame cue / pocket boxes so the finally
                # block can run the end-of-video scratch & off-field checks.
                self._last_pocket_bboxes = [tuple(p.bbox) for p in pockets]
                self._last_cue_bbox = tuple(cue.bbox) if cue is not None else None

                if cue is not None:
                    self.cue_missing_frames = 0
                    self._missing_fired     = False
                    self._off_table_fired   = False
                    cc = cue.center

                    if self.prev_cue_center is not None:
                        self.cue_velocity = pt_dist(cc, self.prev_cue_center)
                    self.prev_cue_center = cc

                    # Shot start
                    if (not self.shot_in_progress
                            and self.cue_velocity > SHOT_START_VEL):
                        self.shot_in_progress     = True
                        self.lowest_at_shot_start = lowest_ball
                        self.checked_contact_tids.clear()
                        self._low_vel_run         = 0
                        self.balls_hit_rail_this_shot.clear()
                        self.post_shot_eval_frame = None
                        self.balls_moved_this_shot.clear()
                        self.trajectory_candidates.clear()
                        self.first_trajectory_hit_fired = False
                        self._no_legal_ball_hit_fired = False
                        
                        _all_start_balls = balls + ([cue] if cue else [])
                        self.ball_start_info = {
                            _b.track_id: {
                                "center": _b.center,
                                "on_rail": self._ball_at_rail(_b.bbox, frame.shape),
                                "cls": _b.stable_cls  # <--- NEW: Save the class
                            } for _b in _all_start_balls
                        }
                        self.balls_left_rail_this_shot.clear()
                        
                        self.is_break_shot = (self.rack_frames_since_seen < 30)
                        
                        log_msg = "Shot started (Break Shot)" if self.is_break_shot else "Shot started"
                        self._add_event_log(log_msg)

                    # First overlap OR trajectory intersection → queue a PendingContact
                    if self.shot_in_progress and not self.contact_occurred_this_shot:
                        for ball in balls:
                            if ball.track_id not in self.checked_contact_tids:
                                
                                is_hit = False
                                
                                # Check 1: Standard Bounding Box overlap
                                if boxes_overlap(cue.bbox, ball.bbox, pad=CONTACT_BBOX_PAD):
                                    is_hit = True
                                    
                                # Check 2: Trajectory Swept-Path (catches high-speed "tunneling")
                                elif self.prev_cue_center is not None and self.cue_velocity > 5.0:
                                    # Calculate distance from object ball to the cue's path this frame
                                    path_dist = pt_to_line_segment_dist(ball.center, self.prev_cue_center, cue.center)
                                    
                                    # If the path passed within ~20 pixels (roughly a ball's radius + padding),
                                    # the cue ball swept through this space.
                                    if path_dist < 20.0: 
                                        is_hit = True
                                
                                # If either check passed, queue the ball to see if it actually moves
                                if is_hit:
                                    self.checked_contact_tids.add(ball.track_id)
                                    self.pending_contacts.append(PendingContact(
                                        contacted_track_id = ball.track_id,
                                        contacted_class    = ball.stable_cls,
                                        pos_at_contact     = ball.center,
                                        created_frame      = self.frame_idx,
                                        shot_target_class  = self.lowest_at_shot_start,
                                    ))
                                
                    # ── NEW: Track the exact frame each object ball first moves ──
                    if self.shot_in_progress:
                        for b in balls: # 'balls' only contains object balls, not the cue
                            if b.track_id not in self.balls_moved_this_shot:
                                start_info = self.ball_start_info.get(b.track_id)
                                if start_info:
                                    dist = pt_dist(b.center, start_info["center"])
                                    if dist >= MOVEMENT_THRESHOLD_PX:
                                        self.balls_moved_this_shot[b.track_id] = self.frame_idx

                    # ── TRAJECTORY RULE: First ball in cue's path to move = first hit ─────
                    # Step 1 — accumulate candidates: any ball that lies within the
                    # corridor of the cue's projected travel line. The corridor is
                    # scale-relative: half the cue ball's bbox height (i.e. its
                    # radius), so it adapts to camera distance/zoom instead of a
                    # fixed pixel value. Once added, a ball stays a candidate for
                    # the rest of the shot ("is OR was in the trajectory").
                    cue_corridor_px = (cue.bbox[3] - cue.bbox[1]) * 0.5
                    cue_hist = self.ball_positions[cue.track_id]
                    lookback = max(1, min(int(self.fps * 0.15), len(cue_hist) - 1))
                    if len(cue_hist) >= 2:
                        past_pos = cue_hist[max(0, len(cue_hist) - lookback - 1)]
                        curr_pos = cue.center
                        tdx = curr_pos[0] - past_pos[0]
                        tdy = curr_pos[1] - past_pos[1]
                        tspeed = (tdx**2 + tdy**2) ** 0.5
                        if tspeed > 3.0:
                            # Project the path forward from the cue's current position
                            extend = TRAJECTORY_LOOKAHEAD_PX / tspeed
                            far_x  = curr_pos[0] + tdx * extend
                            far_y  = curr_pos[1] + tdy * extend
                            for _b in balls:
                                if _b.track_id not in self.trajectory_candidates:
                                    _d = pt_to_line_segment_dist(
                                        _b.center, curr_pos, (far_x, far_y))
                                    if _d < cue_corridor_px:
                                        self.trajectory_candidates.add(_b.track_id)

                    # Step 2 — among accumulated candidates, whoever moved first
                    # (earliest frame recorded in balls_moved_this_shot) is the
                    # ball the cue hit first.
                    if (not self.contact_occurred_this_shot
                            and not self.first_trajectory_hit_fired
                            and self.trajectory_candidates):
                        moved_candidates = {
                            tid: frm
                            for tid, frm in self.balls_moved_this_shot.items()
                            if tid in self.trajectory_candidates
                        }
                        if moved_candidates:
                            first_hit_tid = min(
                                moved_candidates, key=moved_candidates.get)
                            first_hit_tid, _blocked_moved_tid = (
                                self._line_of_sight_first_hit(first_hit_tid))
                            first_hit_cls = self.ball_start_info.get(
                                first_hit_tid, {}).get("cls")
                            if first_hit_cls is not None:
                                self.first_trajectory_hit_fired  = True
                                self.contact_occurred_this_shot  = True
                                self.post_contact_rail_hit        = False
                                self.post_contact_pocketed        = False
                                self.active_ball_ids_at_contact  = {
                                    _b.track_id: _b.stable_cls for _b in balls}
                                hit_name = CLASS_NAMES.get(
                                    first_hit_cls, f"#{first_hit_tid}")
                                self._add_event_log(
                                    f"Traj 1st hit: {hit_name}")
                                if not self.is_break_shot:
                                    target = self.lowest_at_shot_start
                                    if (target is not None
                                            and first_hit_cls != target):
                                        self._register_no_legal_ball_hit()
                    # ─────────────────────────────────────────────────────────

                    # Shot end
                    if self.shot_in_progress:
                        if self.cue_velocity < SHOT_END_VEL:
                            self._low_vel_run += 1
                            if self._low_vel_run >= shot_end_frames:
                                self.shot_in_progress = False
                                self._low_vel_run     = 0
                                self._add_event_log("Shot ended")
                                # FIX 2: Defer rail evaluation by POST_SHOT_GRACE_FRAMES so
                                # balls still rolling toward the rail are captured first.
                                self.post_shot_eval_frame = (
                                    self.frame_idx + POST_SHOT_GRACE_FRAMES)
                        else:
                            self._low_vel_run = 0

                    # Scratch
                    in_pocket = any(
                        boxes_overlap(cue.bbox, p.bbox) for p in pockets)
                    if in_pocket and self.cue_velocity < SHOT_END_VEL:
                        self.cue_in_pocket_frames += 1
                        if (self.cue_in_pocket_frames >= scratch_frames
                                and not self._scratch_fired):
                            self._register_foul(
                                "Scratch: cue ball resting in pocket")
                            self._scratch_fired = True
                    else:
                        self.cue_in_pocket_frames = 0
                        self._scratch_fired       = False

                    # FIX 3: Use a streak counter so brief pocket-detection gaps
                    # don't break the "cue was near pocket" chain.  If the cue ball
                    # overlaps a pocket this frame, arm the streak for 8 frames;
                    # otherwise count down so it expires naturally.
                    if in_pocket:
                        self.cue_near_pocket_streak = 8
                    else:
                        self.cue_near_pocket_streak = max(
                            0, self.cue_near_pocket_streak - 1)

                    self.cue_was_near_pocket          = (
                        self.cue_near_pocket_streak > 0)
                    self.cue_gone_after_pocket_frames = 0
                    self._pocket_disappear_fired      = False

                else:
                    self.cue_velocity         = 0.0
                    self.cue_in_pocket_frames = 0
                    self.cue_missing_frames  += 1

                    # FIX 3: Fallback — if pocket detection was unreliable but the
                    # cue ball's last known position was close to a stable pocket,
                    # treat it as near-pocket so the disappear-scratch can still fire.
                    if (not self.cue_was_near_pocket
                            and self.prev_cue_center is not None
                            and self.stable_pockets):
                        for _px, _py in self.stable_pockets:
                            if pt_dist(self.prev_cue_center, (_px, _py)) < 70:
                                self.cue_was_near_pocket = True
                                break

                    if self.cue_was_near_pocket:
                        # Disappeared AT a pocket -> treat as a scratch.
                        self.cue_gone_after_pocket_frames += 1
                        if (self.cue_gone_after_pocket_frames >= pocket_disappear_thresh
                                and not self._pocket_disappear_fired):
                            self._register_foul(
                                "Scratch: cue ball entered pocket")
                            self._pocket_disappear_fired = True
                            self._add_event_log("Cue ball scratch")

                    elif self.prev_cue_center is not None:
                        # Cue was seen earlier and has now vanished AWAY from any
                        # pocket. If it stays gone past the grace window, the cue
                        # ball has left the playing field -> off-the-table foul.
                        if (self.cue_missing_frames >= off_table_thresh
                                and not self._off_table_fired):
                            self._register_foul(
                                "Cue ball off the table")
                            self._off_table_fired = True
                            self._add_event_log("Cue ball off table")

                # ── 4b. Deferred post-shot rail evaluation ─────────────────────────────
                if (self.post_shot_eval_frame is not None
                        and self.frame_idx >= self.post_shot_eval_frame):
                    
                    # --- NEW: First-Mover Fallback Hit Detection ---
                    if not self.contact_occurred_this_shot and self.balls_moved_this_shot:
                        # Find the ball track_id that moved on the earliest frame
                        first_moved_id = min(self.balls_moved_this_shot, key=self.balls_moved_this_shot.get)
                        first_moved_id, blocked_moved_id = (
                            self._line_of_sight_first_hit(first_moved_id))
                        
                        # Retrieve what class that ball was at the start of the shot
                        if first_moved_id in self.ball_start_info:
                            first_moved_cls = self.ball_start_info[first_moved_id]["cls"]
                            
                            self.contact_occurred_this_shot = True
                            hit_name = CLASS_NAMES.get(first_moved_cls, f"#{first_moved_id}")
                            if blocked_moved_id is not None:
                                self._add_event_log(
                                    f"Fallback hit: {hit_name} blocked line of sight")
                            else:
                                self._add_event_log(
                                    f"Fallback hit: {hit_name} moved first")
                            
                            # Standard legal-ball foul check
                            if not self.is_break_shot:
                                target = self.lowest_at_shot_start
                                if target is not None and first_moved_cls != target:
                                    self._register_no_legal_ball_hit()
                    if not self.contact_occurred_this_shot:
                        self._register_no_legal_ball_hit()
                    # ---------------------------------------------------------------

                    self._evaluate_rail_checks()
                    self.contact_occurred_this_shot = False
                    self.is_break_shot              = False
                    self.post_shot_eval_frame       = None

                # (c) Resolve pending contacts.
                # Register the contact the moment the struck ball has clearly
                # moved, instead of waiting the full MOVEMENT_CONFIRM_FRAMES
                # window. This stamps the contact close to the real touch frame,
                # so timing-dependent checks (e.g. the post-contact rail flag)
                # line up with reality instead of lagging behind it. The window
                # now only acts as a no-movement timeout that rejects mere
                # occlusion — the cue passing over a ball that never moves.
                still_pending: List[PendingContact] = []
                for pc in self.pending_contacts:
                    age = self.frame_idx - pc.created_frame

                    if pc.contacted_track_id in ball_by_id:
                        cur_pos = ball_by_id[pc.contacted_track_id].center
                        moved   = pt_dist(cur_pos, pc.pos_at_contact)
                    else:
                        moved = 0.0

                    if moved >= MOVEMENT_THRESHOLD_PX:
                        # Confirmed contact (resolved — not re-queued).
                        if not self.contact_occurred_this_shot:
                            target   = pc.shot_target_class
                            hit_name = CLASS_NAMES[pc.contacted_class]

                            self._add_event_log(f"Cue hit {hit_name}")

                            # Legal-ball foul (EXEMPT if it's a break shot)
                            if not self.is_break_shot:
                                if target is not None and pc.contacted_class != target:
                                    self._register_no_legal_ball_hit()

                            self.contact_occurred_this_shot = True
                            self.post_contact_rail_hit = False
                            self.post_contact_pocketed = False
                            self.active_ball_ids_at_contact = {
                                b.track_id: b.stable_cls for b in balls
                            }
                    elif age < MOVEMENT_CONFIRM_FRAMES:
                        still_pending.append(pc)   # keep watching until timeout
                    else:
                        # Timed out with no movement -> treat as occlusion, drop.
                        self.checked_contact_tids.discard(pc.contacted_track_id)

                self.pending_contacts = still_pending

                # ── 5. Draw & write output ────────────────────────────────
                annotated = self._draw(
                    frame, balls, cue, pockets, cue_sticks, lowest_ball)

                if self.video_writer:
                    self.video_writer.write(annotated)
                self._last_annotated = annotated
                self._display_frame(annotated)

                time.sleep(max(0.0, 1.0 / self.fps - 0.015))

        finally:
            self.is_playing  = False
            _eov_fouls_before = self._foul_count
            if self.shot_in_progress or self.post_shot_eval_frame is not None:
                if self.contact_occurred_this_shot:
                    self._evaluate_rail_checks()
                else:
                    self._register_no_legal_ball_hit()

            # End-of-video scratch: if the cue ball's whole bounding box is
            # inside a pocket's bounding box on the final frame, it was potted.
            if (not self._scratch_fired and not self._pocket_disappear_fired
                    and not self._cue_pocket_replace_fired
                    and self._last_cue_bbox is not None
                    and any(bbox_inside(self._last_cue_bbox, pb)
                            for pb in self._last_pocket_bboxes)):
                self._register_foul("Scratch: cue ball resting in pocket")
                self._scratch_fired = True
                self._add_event_log("Cue ball scratch")

            # End-of-video cue-off-field check.  When processing ends (video
            # finished OR Stop pressed), if the cue ball is not present on the
            # final processed frame and was NOT pocketed (no scratch logged),
            # it has left the playing field -> off-the-table foul.  Fires even
            # if the cue was never detected, and dedupes against an off-table
            # foul that may have already fired mid-video.
            cue_pocketed = (self._scratch_fired or self._pocket_disappear_fired
                            or self._cue_pocket_replace_fired)
            if (self.cue_missing_frames > 0
                    and not cue_pocketed
                    and not self._off_table_fired):
                self._register_foul("Cue ball off the table")
                self._off_table_fired = True
                self._add_event_log("Cue ball off table")

            # The end-of-video fouls above are registered after the final frame
            # was drawn, so their banner never reached the screen/output video.
            # Stamp the active banner onto the last frame and push it out so the
            # red label actually appears.
            if (self._foul_count > _eov_fouls_before
                    and self._last_annotated is not None):
                annotated = self._render_foul_banner(self._last_annotated.copy())
                self._last_annotated = annotated
                self._display_frame(annotated)
                if self.video_writer:
                    for _ in range(max(1, int(FOUL_BANNER_SECS * self.fps))):
                        self.video_writer.write(annotated)

            self.root.after(0, self._cleanup)

    # ─────────────────────────── Drawing ─────────────────────────────────

    def _draw(self, frame, balls: List[Track], cue: Optional[Track],
              pockets: List[Track], cue_sticks: List[Track],
              lowest_ball: Optional[int]):

        out = frame.copy()
        h, w = out.shape[:2]

        if self.rail_polygon is not None:
            cv2.polylines(out, [self.rail_polygon], isClosed=True,
                          color=(0, 160, 160), thickness=2)
            _short = {
                "top_left": "TL", "top_middle": "TM", "top_right": "TR",
                "bottom_left": "BL", "bottom_middle": "BM", "bottom_right": "BR",
            }
            for _name, (_px, _py) in self.pocket_positions.items():
                cv2.circle(out, (int(_px), int(_py)), 4, (0, 200, 200), -1)
                cv2.putText(out, _short.get(_name, _name),
                            (int(_px) + 6, int(_py) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)
            cv2.putText(out, "rail",
                        (int(self.rail_polygon[0][0]) + 4,
                         int(self.rail_polygon[0][1]) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 160, 160), 1)
        elif self.table_bounds is not None:
            rx1, ry1, rx2, ry2 = (int(v) for v in self.table_bounds)
            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 160, 160), 1)
            m = RAIL_MARGIN_PX
            cv2.rectangle(out,
                          (rx1 + m, ry1 + m),
                          (rx2 - m, ry2 - m),
                          (0, 100, 100), 1)
            cv2.putText(out, "rail zone (forming...)", (rx1 + 4, ry1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 160, 160), 1)

        needs_rail = (self.contact_occurred_this_shot and 
                      not self.post_contact_rail_hit and 
                      not self.post_contact_pocketed)

        for p in pockets:
            x1, y1, x2, y2 = map(int, p.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 180), 2)
            cv2.putText(out, "pocket", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 180), 1)

        for s in cue_sticks:
            x1, y1, x2, y2 = map(int, s.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 140, 255), 2)

        for b in balls:
            x1, y1, x2, y2 = map(int, b.bbox)
            is_target       = (b.stable_cls == lowest_ball)
            is_waiting_rail = (needs_rail and is_target)

            if is_waiting_rail:
                color, thick = (0, 140, 255), 3
            elif is_target:
                color, thick = (0, 255, 60), 3
            else:
                color, thick = (0, 215, 215), 2

            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
            
            label = CLASS_NAMES.get(b.stable_cls, f"#{b.track_id}")
            if is_waiting_rail:
                label += " ⚠ rail?"
            label += f"  #{b.track_id}"
            
            cv2.putText(out, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        if cue is not None:
            x1, y1, x2, y2 = map(int, cue.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(out, f"cue  #{cue.track_id}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            # ─── NEW: Cue Ball Trajectory Prediction ─────────────────────
            history = self.ball_positions[cue.track_id]
            
            # Look back ~0.5 seconds for the past position. 
            # (Using 0.5s is usually better than 1s in billiards so the line doesn't 
            # calculate vectors from before it bounced off a rail, but you can change 
            # the 0.5 to 1.0 if you strictly want a full second ago).
            lookback_frames = int(self.fps * 0.2) 
            
            if len(history) >= 2:
                past_idx = max(0, len(history) - lookback_frames - 1)
                past_center = history[past_idx]
                curr_center = cue.center
                
                # Vector math: Where is it now minus where was it?
                dx = curr_center[0] - past_center[0]
                dy = curr_center[1] - past_center[1]
                
                # Calculate distance moved in that time window
                dist = (dx**2 + dy**2)**0.5
                
                # Only draw the trajectory if it's actually moving (prevents jitter when still)
                if dist > 5.0:
                    # Extrapolate forward. Scale determines how long the line stretches.
                    # scale = 2.0 means "draw the line twice as far as it traveled in the lookback window"
                    scale = 2.5 
                    
                    pred_x = int(curr_center[0] + dx * scale)
                    pred_y = int(curr_center[1] + dy * scale)
                    
                    # Draw the predicted trajectory line (Cyan color)
                    cv2.line(out, (int(curr_center[0]), int(curr_center[1])), 
                                  (pred_x, pred_y), (255, 255, 0), 2)
                    
                    # Draw a small red target dot at the predicted destination
                    cv2.circle(out, (pred_x, pred_y), 4, (0, 0, 255), -1)
            # ─────────────────────────────────────────────────────────────

        # Panel sized for the lowest-ball header plus the event log
        panel_bottom = 120 + (len(self.event_log) * 25)

        ov = out.copy()
        cv2.rectangle(ov, (8, 8), (600, panel_bottom), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, out, 0.45, 0, out)
        cv2.rectangle(out, (8, 8), (600, panel_bottom), (0, 200, 200), 2)

        cv2.putText(out, "LOWEST BALL ON TABLE",
                    (22, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (160, 200, 200), 2)

        if lowest_ball is not None:
            label    = CLASS_NAMES[lowest_ball]
            col_name = BALL_COLOR_ID.get(lowest_ball, "")
            swatch   = BALL_BGR.get(lowest_ball)
            sx, sy, sr = 34, 74, 16

            # Colour swatch of the ball's real colour (two-tone for stripes)
            if isinstance(swatch, tuple) and swatch and isinstance(swatch[0], tuple):
                cv2.ellipse(out, (sx, sy), (sr, sr), 0, 180, 360, swatch[0], -1)
                cv2.ellipse(out, (sx, sy), (sr, sr), 0,   0, 180, swatch[1], -1)
            elif swatch is not None:
                cv2.circle(out, (sx, sy), sr, swatch, -1)
            cv2.circle(out, (sx, sy), sr, (255, 255, 255), 2)

            text = f"{label} [{col_name}]" if col_name else label
            cv2.putText(out, text, (sx + sr + 14, sy + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        else:
            cv2.putText(out, "NONE DETECTED", (22, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (130, 130, 130), 3)

        # ─── Event log ──────────────────────────────────────────────────────────
        y_offset = 125
        for event_str in self.event_log:
            cv2.putText(out, event_str, (22, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)
            y_offset += 25

        # ═══ DEBUG_CONTACT overlay — delete this whole block to remove ═══════════
        if SHOW_CONTACT_DEBUG:
            def _names(ids):
                if not ids:
                    return "-"
                return ", ".join(CLASS_NAMES.get(i, f"#{i}") for i in sorted(ids))

            dbg_lines = [
                f"shot={self.shot_in_progress}  contact={self.contact_occurred_this_shot}",
                f"traj_cand: {_names(self.trajectory_candidates)}",
                f"moved:     {_names(self.balls_moved_this_shot.keys())}",
                f"potted:    {_names(self.potted_balls)}",
            ]
            dh = out.shape[0]
            y0 = dh - 18 - (len(dbg_lines) - 1) * 22
            for i, line in enumerate(dbg_lines):
                y = y0 + i * 22
                cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 3)          # black outline for legibility
                cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (80, 255, 255), 1)     # yellow text
        # ═══ end DEBUG_CONTACT overlay ═══════════════════════════════════════════

        out = self._render_foul_banner(out)

        return out

    def _render_foul_banner(self, out):
        """Draw the active red FOUL banner onto `out`, auto-fit to frame width.

        Extracted so the end-of-video fouls (registered in the finally block,
        after the last _draw) can also stamp their banner onto the final frame.
        """
        if not (self.foul_banner_text
                and self.frame_idx < self.foul_banner_until_frame):
            return out

        w      = out.shape[1]
        banner = f"  FOUL: {self.foul_banner_text}  "
        font   = cv2.FONT_HERSHEY_SIMPLEX
        thick  = 2

        # Shrink the font until the banner fits inside the frame width, so long
        # foul reasons are never drawn off-screen.
        scale = 0.85
        (tw, th), _ = cv2.getTextSize(banner, font, scale, thick)
        while tw > (w - 20) and scale > 0.4:
            scale -= 0.05
            (tw, th), _ = cv2.getTextSize(banner, font, scale, thick)

        by = 20
        bx = max(10, w - tw - 10)          # right-aligned, always on-screen
        cv2.rectangle(out, (bx, by),
                      (bx + tw, by + th + 20), (0, 0, 190), -1)
        cv2.rectangle(out, (bx, by),
                      (bx + tw, by + th + 20), (80, 80, 255), 2)
        cv2.putText(out, banner, (bx, by + th + 7),
                    font, scale, (255, 255, 255), thick)
        return out

    # ──────────────────────── Utilities ──────────────────────────────────

    def _register_foul(self, reason: str):
        t      = self.frame_idx / max(self.fps, 1e-6)
        ts_str = f"[{int(t // 60):02d}:{int(t % 60):02d}]"
        entry  = f"[{ts_str}]  {reason}"

        self._foul_count            += 1
        self.foul_banner_text        = reason
        self.foul_banner_until_frame = (
            self.frame_idx + int(FOUL_BANNER_SECS * self.fps))

        if self.foul_log:
            self.foul_log.write(entry + "\n")
            self.foul_log.flush()

        def _add():
            self.foul_listbox.insert(tk.END, entry)
            self.foul_listbox.see(tk.END)
        self.root.after(0, _add)

    def _display_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        max_w = 1200
        if w > max_w:
            scale     = max_w / w
            frame_bgr = cv2.resize(frame_bgr, (max_w, int(h * scale)))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        def _update():
            img   = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_lbl.imgtk = imgtk
            self.video_lbl.config(image=imgtk)
        self.root.after(0, _update)


# ═══════════════════════════ Entry point ══════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app = FoulDetectorApp(root)

    def _on_close():
        # Program is dying -> stop any running recording/detection cleanly.
        try:
            app.is_recording = False
            app.is_playing = False
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()