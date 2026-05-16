"""
Gesture Glove — Main GUI Application
3-tab CustomTkinter app: Recorder, Train & Export, Live Debug
"""
import customtkinter as ctk
from recorder import SerialManager
from gui_recorder import RecorderTab
from gui_train import TrainTab
from gui_debug import DebugTab


class GestureGloveApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ── Window Setup ──
        self.title("🧤 Gesture Glove — ML Pipeline")
        self.geometry("1280x780")
        self.minsize(1000, 650)

        # Theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Shared serial manager (used by Recorder and Debug tabs)
        self.serial = SerialManager()

        # ── Header ──
        header = ctk.CTkFrame(self, height=50, corner_radius=0, fg_color="#0f0f23")
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(header, text="🧤 Gesture Glove", font=("Inter", 20, "bold"),
                     text_color="#E0E7FF").pack(side="left", padx=16)
        ctk.CTkLabel(header, text="ESP32-S3 + MPU6050 → USB HID Keyboard",
                     font=("Inter", 11), text_color="#6B7280").pack(side="left", padx=8)

        # ── Tabview ──
        self.tabview = ctk.CTkTabview(self, corner_radius=12, fg_color="#1a1a2e",
                                       segmented_button_fg_color="#16213e",
                                       segmented_button_selected_color="#8B5CF6",
                                       segmented_button_unselected_color="#374151")
        self.tabview.pack(fill="both", expand=True, padx=12, pady=(6, 12))

        # Create tabs
        tab1 = self.tabview.add("📡 Recorder")
        tab2 = self.tabview.add("🧠 Train & Export")
        tab3 = self.tabview.add("🔍 Live Debug")

        # Build tab contents
        self.recorder_tab = RecorderTab(tab1, self.serial)
        self.recorder_tab.pack(fill="both", expand=True)
        # recorder_tab handles keyboard hold/release via callbacks below

        self.train_tab = TrainTab(tab2)
        self.train_tab.pack(fill="both", expand=True)

        self.debug_tab = DebugTab(tab3, self.serial)
        self.debug_tab.pack(fill="both", expand=True)

        # ── Global keyboard — hold 1-5 to record ──────────────
        self.bind("<KeyPress>", self._on_key_press)
        self.bind("<KeyRelease>", self._on_key_release)

        # ── Footer Status Bar ──
        footer = ctk.CTkFrame(self, height=28, corner_radius=0, fg_color="#0f0f23")
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        ctk.CTkLabel(footer, text="v1.0 | XIAO Seeed ESP32-S3 | MPU6050 @ 0x69",
                     font=("Inter", 10), text_color="#4B5563").pack(side="left", padx=12)
        self.fps_label = ctk.CTkLabel(footer, text="", font=("Inter", 10), text_color="#4B5563")
        self.fps_label.pack(side="right", padx=12)

        # Handle close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_key_press(self, event):
        self.recorder_tab.on_key_press(event.keysym)

    def _on_key_release(self, event):
        self.recorder_tab.on_key_release(event.keysym)

    def _on_close(self):
        if self.serial.is_connected:
            self.serial.disconnect()
        self.destroy()


def main():
    app = GestureGloveApp()
    app.mainloop()


if __name__ == "__main__":
    main()
