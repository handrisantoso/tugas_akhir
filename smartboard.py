"""
smartboard.py — Smartboard ringan untuk Gesture Glove.

Terhubung ke ESP32 (firmware *_ble) lewat BLE, menerima prediksi gesture,
lalu menekan tombol sistem ke aplikasi aktif (PowerPoint/PDF/dll):

    flick_up    → F5     (mulai presentasi)
    wave_left   → ←      (slide sebelumnya)
    wave_right  → →      (slide berikutnya)
    flick_down  → Esc    (keluar presentasi)
    idle        → (diam)

Jalankan:  python smartboard.py
"""

import time
import threading
import customtkinter as ctk

from ble_manager import BLEManager

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    HAVE_PYAUTOGUI = True
except Exception:
    HAVE_PYAUTOGUI = False

# gesture → (tombol pyautogui, label aksi)
GESTURE_ACTIONS = {
    "flick_up":   ("f5",    "▶  Mulai presentasi (F5)"),
    "wave_left":  ("left",  "◀  Slide sebelumnya (←)"),
    "wave_right": ("right", "▶  Slide berikutnya (→)"),
    "flick_down": ("esc",   "⏹  Keluar presentasi (Esc)"),
}
CLIENT_DEBOUNCE_MS = 600   # abaikan aksi berturut yang terlalu cepat


class Smartboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🎯 Gesture Smartboard")
        self.geometry("560x600")
        self.minsize(480, 560)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.ble = BLEManager()
        self.ble.on_prediction = self._on_prediction
        self.ble.on_status_change = self._on_status
        self._ble_targets = {}        # label combobox -> address
        self._last_action_ms = 0

        # ── Header ──
        ctk.CTkLabel(self, text="🎯 Gesture Smartboard", font=("Inter", 22, "bold")).pack(pady=(14, 2))
        ctk.CTkLabel(self, text="Kontrol presentasi via gesture glove (BLE)",
                     font=("Inter", 11), text_color="#9CA3AF").pack()

        if not HAVE_PYAUTOGUI:
            ctk.CTkLabel(self, text="⚠ pyautogui tidak terpasang — aksi tombol nonaktif. "
                                    "pip install pyautogui", text_color="#F59E0B",
                         font=("Inter", 11)).pack(pady=4)

        # ── Connection ──
        conn = ctk.CTkFrame(self, corner_radius=10)
        conn.pack(fill="x", padx=16, pady=(12, 6))
        row = ctk.CTkFrame(conn, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=10)
        self.dev_var = ctk.StringVar()
        self.dev_menu = ctk.CTkComboBox(row, variable=self.dev_var, width=240, state="readonly")
        self.dev_menu.pack(side="left", padx=(0, 6))
        self.scan_btn = ctk.CTkButton(row, text="🔵 Scan", width=80, command=self.scan_ble,
                                      fg_color="#3B82F6", hover_color="#2563EB")
        self.scan_btn.pack(side="left", padx=2)
        self.conn_btn = ctk.CTkButton(row, text="Connect", width=90, command=self.toggle_connection,
                                      fg_color="#22C55E", hover_color="#16A34A")
        self.conn_btn.pack(side="left", padx=2)
        self.status_lbl = ctk.CTkLabel(conn, text="● Disconnected", text_color="#EF4444", font=("Inter", 12))
        self.status_lbl.pack(anchor="w", padx=12, pady=(0, 8))

        # ── Enable actions switch ──
        self.enabled = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(self, text="Aktifkan aksi tombol (kirim ke aplikasi aktif)",
                      variable=self.enabled, onvalue=True, offvalue=False).pack(pady=(2, 8))

        # ── Last gesture / action ──
        big = ctk.CTkFrame(self, corner_radius=10, fg_color="#16213e")
        big.pack(fill="x", padx=16, pady=6)
        self.gesture_lbl = ctk.CTkLabel(big, text="—", font=("Inter", 30, "bold"), text_color="#E0E7FF")
        self.gesture_lbl.pack(pady=(16, 2))
        self.action_lbl = ctk.CTkLabel(big, text="Menunggu gesture…", font=("Inter", 14), text_color="#9CA3AF")
        self.action_lbl.pack(pady=(0, 16))

        # ── Mapping table ──
        mp = ctk.CTkFrame(self, corner_radius=10)
        mp.pack(fill="x", padx=16, pady=6)
        ctk.CTkLabel(mp, text="Pemetaan Gesture → Aksi", font=("Inter", 13, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
        for g, (key, label) in GESTURE_ACTIONS.items():
            ctk.CTkLabel(mp, text=f"   {g:<12} →  {label}", font=("Consolas", 12),
                         text_color="#D1D5DB").pack(anchor="w", padx=10)
        ctk.CTkLabel(mp, text="   idle         →  (tidak ada aksi)", font=("Consolas", 12),
                     text_color="#6B7280").pack(anchor="w", padx=10, pady=(0, 8))

        # ── Log ──
        self.log_box = ctk.CTkTextbox(self, height=120, font=("Consolas", 11))
        self.log_box.pack(fill="both", expand=True, padx=16, pady=(6, 12))
        self.log_box.configure(state="disabled")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Logging ──
    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{ts}] {msg}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    # ── BLE scan ──
    def scan_ble(self):
        self.scan_btn.configure(text="…", state="disabled")
        self.log("Memindai BLE (±6 detik)…")

        def worker():
            err, devices = None, []
            try:
                devices = BLEManager.scan(timeout=6.0)
            except Exception as e:
                err = str(e)
            self.after(0, lambda: self._scan_done(devices, err))

        threading.Thread(target=worker, daemon=True).start()

    def _scan_done(self, devices, err):
        self.scan_btn.configure(text="🔵 Scan", state="normal")
        if err:
            self.log(f"Scan error: {err}")
            return
        if not devices:
            self.log("Tidak ada perangkat 'GestureGlove'.")
            return
        labels = []
        for addr, label in devices:
            self._ble_targets[label] = addr
            labels.append(label)
        self.dev_menu.configure(values=labels)
        self.dev_var.set(labels[0])
        self.log(f"Ditemukan {len(labels)} perangkat. Pilih lalu Connect.")

    # ── BLE connect/disconnect (threaded, GUI tidak freeze) ──
    def toggle_connection(self):
        if self.ble.is_connected:
            self.conn_btn.configure(state="disabled", text="…")
            def worker():
                try: self.ble.disconnect()
                except Exception: pass
                self.after(0, self._after_disconnect)
            threading.Thread(target=worker, daemon=True).start()
            return

        label = self.dev_var.get()
        addr = self._ble_targets.get(label)
        if not addr:
            self.log("Pilih perangkat dulu (Scan).")
            return
        self.conn_btn.configure(state="disabled", text="…")
        self.status_lbl.configure(text="● Connecting…", text_color="#F59E0B")
        self.log(f"Connecting {addr} …")

        def worker():
            err = None
            try:
                self.ble.connect(addr)
            except Exception as e:
                err = str(e)
            self.after(0, lambda: self._after_connect(label, err))

        threading.Thread(target=worker, daemon=True).start()

    def _after_connect(self, label, err):
        self.conn_btn.configure(state="normal")
        if err:
            self.conn_btn.configure(text="Connect", fg_color="#22C55E", hover_color="#16A34A")
            self.status_lbl.configure(text="● Error", text_color="#EF4444")
            self.log(f"ERROR: {err}")
            return
        self.conn_btn.configure(text="Disconnect", fg_color="#EF4444", hover_color="#DC2626")
        self.status_lbl.configure(text="● Connected", text_color="#22C55E")
        self.log(f"Terhubung ke {label}. Lakukan gesture!")

    def _after_disconnect(self):
        self.conn_btn.configure(state="normal", text="Connect",
                                fg_color="#22C55E", hover_color="#16A34A")
        self.status_lbl.configure(text="● Disconnected", text_color="#EF4444")
        self.log("Terputus.")

    # ── Callback dari BLE (dipanggil di thread loop BLE) ──
    def _on_prediction(self, gesture, confidence, latency):
        self.after(0, lambda: self._handle(gesture, confidence, latency))

    def _on_status(self, status):
        # Putus tak terduga
        if status in ("disconnected", "error"):
            self.after(0, self._after_disconnect)

    def _handle(self, gesture, confidence, latency):
        self.gesture_lbl.configure(text=gesture)
        action = GESTURE_ACTIONS.get(gesture)
        if action is None:
            self.action_lbl.configure(text="(idle / tidak ada aksi)", text_color="#6B7280")
            self.log(f"{gesture}  conf={confidence:.2f}  (tidak ada aksi)")
            return

        key, label = action
        self.action_lbl.configure(text=label, text_color="#22C55E")

        now_ms = time.time() * 1000
        if now_ms - self._last_action_ms < CLIENT_DEBOUNCE_MS:
            self.log(f"{gesture}  conf={confidence:.2f}  (di-skip, terlalu cepat)")
            return
        self._last_action_ms = now_ms

        if not self.enabled.get():
            self.log(f"{gesture} → {key.upper()}  (aksi NONAKTIF)")
            return

        if HAVE_PYAUTOGUI:
            try:
                pyautogui.press(key)
                self.log(f"{gesture}  conf={confidence:.2f} → tekan {key.upper()}")
            except Exception as e:
                self.log(f"Gagal tekan {key}: {e}")
        else:
            self.log(f"{gesture} → {key.upper()}  (pyautogui tidak ada)")

    def _on_close(self):
        try:
            if self.ble.is_connected:
                self.ble.disconnect()
        except Exception:
            pass
        self.destroy()


def main():
    Smartboard().mainloop()


if __name__ == "__main__":
    main()
