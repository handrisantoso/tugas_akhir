"""
ble_manager.py — koneksi NIRKABEL ke Gesture Glove via BLE (Nordic UART Service).

Drop-in pengganti SerialManager untuk perekaman raw IMU tanpa kabel USB.
API-nya SENGAJA sama dengan SerialManager (recorder.py) supaya GestureRecorder &
GUI bisa memakainya tanpa perubahan logika:
    - properti  : is_connected
    - metode    : connect(address), disconnect(), clear_queue(), get_imu_sample(timeout), send_command(cmd)
    - callback  : on_imu_data, on_prediction, on_raw_line, on_status_change
    - statis    : scan(timeout) -> [(address, label)]  (mirip list_ports())

bleak bersifat async; di sini sebuah event loop asyncio dijalankan di thread
daemon terpisah, dan panggilan sinkron dijembatani via run_coroutine_threadsafe.
"""

import asyncio
import threading
import queue

from config import NUM_AXES

try:
    from bleak import BleakScanner, BleakClient
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Library 'bleak' belum terpasang. Jalankan: pip install bleak"
    ) from e

# Nordic UART Service (cocok dengan firmware gesture_glove_esp32.ino)
NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_TX_UUID      = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # device → host (notify)
TARGET_NAME      = "GestureGlove"

# ─── Event loop bersama (satu untuk seluruh aplikasi) ───────────
_loop = None
_loop_thread = None


def _ensure_loop():
    global _loop, _loop_thread
    if _loop is None:
        _loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(target=_loop.run_forever, daemon=True)
        _loop_thread.start()
    return _loop


def _run(coro, timeout=None):
    """Jalankan coroutine di loop bersama, blok sampai selesai."""
    fut = asyncio.run_coroutine_threadsafe(coro, _ensure_loop())
    return fut.result(timeout)


class BLEManager:
    """Manajer koneksi BLE thread-safe untuk ESP32-S3 (Nordic UART Service)."""

    def __init__(self):
        # Callbacks (sama persis dengan SerialManager)
        self.on_imu_data = None       # callback(ax, ay, az, gx, gy, gz)
        self.on_prediction = None     # callback(gesture_name, confidence, latency_ms)
        self.on_raw_line = None       # callback(line_str)
        self.on_status_change = None  # callback(status_str)

        self._data_queue = queue.Queue()
        self._client = None
        self._connected = False
        self._rx_buf = bytearray()    # buffer untuk merakit baris (split '\n')

    # ─── Penemuan perangkat ─────────────────────────────────────
    @staticmethod
    def scan(timeout=6.0):
        """Pindai perangkat BLE bernama 'GestureGlove' / beriklan NUS.

        Returns list of (address, label) — label siap tampil di combobox.
        Aman dipanggil dari thread mana pun (mengeksekusi di loop bersama).
        """
        async def _scan():
            found = []
            try:
                results = await BleakScanner.discover(timeout=timeout, return_adv=True)
                items = results.values()  # dict {addr: (device, adv)}
            except TypeError:
                # bleak lama: discover() -> list[BLEDevice]
                devices = await BleakScanner.discover(timeout=timeout)
                items = [(d, None) for d in devices]

            for dev, adv in items:
                name = (getattr(dev, "name", None)
                        or (getattr(adv, "local_name", None) if adv else None) or "")
                uuids = [u.lower() for u in (getattr(adv, "service_uuids", None) or [])] if adv else []
                if TARGET_NAME.lower() in name.lower() or NUS_SERVICE_UUID in uuids:
                    label = f"BLE: {name or TARGET_NAME} ({dev.address})"
                    found.append((dev.address, label))
            return found

        try:
            return _run(_scan(), timeout=timeout + 5)
        except Exception:
            return []

    # Alias agar setara dengan SerialManager.list_ports()
    @staticmethod
    def list_ports():
        return BLEManager.scan()

    # ─── Status ─────────────────────────────────────────────────
    @property
    def is_connected(self):
        return self._connected and self._client is not None

    # ─── Koneksi ────────────────────────────────────────────────
    def connect(self, address, baud=None):
        """Hubungkan ke perangkat BLE (baud diabaikan, demi parity API).

        Di Windows, connect via string alamat sering menggantung; pola andal =
        temukan BLEDevice dulu (find_device_by_address), baru BleakClient(device).
        """
        async def _connect():
            device = await BleakScanner.find_device_by_address(address, timeout=12.0)
            if device is None:
                raise ConnectionError(
                    "perangkat tidak ditemukan saat connect — dekatkan glove, "
                    "pastikan masih advertising, lalu scan ulang (🔵 BLE)")
            # use_cached_services=False -> abaikan service HID lama yg di-cache
            # Windows dari firmware "GestureGlove" sebelumnya (paksa discovery fresh).
            client = BleakClient(device, disconnected_callback=self._on_disconnected,
                                 timeout=20.0, winrt={"use_cached_services": False})
            await client.connect()
            await client.start_notify(NUS_TX_UUID, self._notify_handler)
            return client

        try:
            self._client = _run(_connect(), timeout=40)
        except Exception as e:
            self._client = None
            self._connected = False
            msg = str(e) or type(e).__name__   # TimeoutError dll. punya str kosong
            raise ConnectionError(f"Gagal connect BLE {address}: {msg}")

        self._connected = True
        self._rx_buf.clear()
        if self.on_status_change:
            self.on_status_change("connected")

    def disconnect(self):
        async def _disc():
            if self._client:
                try:
                    await self._client.stop_notify(NUS_TX_UUID)
                except Exception:
                    pass
                await self._client.disconnect()

        try:
            _run(_disc(), timeout=10)
        except Exception:
            pass
        self._client = None
        self._connected = False
        if self.on_status_change:
            self.on_status_change("disconnected")

    def _on_disconnected(self, client):
        """Dipanggil bleak bila link putus tak terduga."""
        self._connected = False
        if self.on_status_change:
            self.on_status_change("disconnected")

    def send_command(self, cmd):
        """Parity dgn SerialManager. Firmware perekam hanya punya TX (notify),
        tidak ada karakteristik RX — jadi ini no-op."""
        return

    # ─── Antrian data (dipakai GestureRecorder) ─────────────────
    def get_imu_sample(self, timeout=0.05):
        """Ambil satu sampel IMU dari antrian. (ax,ay,az,gx,gy,gz) atau None."""
        try:
            return self._data_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self):
        while not self._data_queue.empty():
            try:
                self._data_queue.get_nowait()
            except queue.Empty:
                break

    # ─── Penerimaan notifikasi BLE ──────────────────────────────
    def _notify_handler(self, sender, data: bytearray):
        """Rakit byte menjadi baris (pisah '\n'), lalu parse seperti serial."""
        self._rx_buf.extend(data)
        while b"\n" in self._rx_buf:
            idx = self._rx_buf.index(b"\n")
            raw = bytes(self._rx_buf[:idx])
            del self._rx_buf[:idx + 1]
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                self._process_line(line)

    def _process_line(self, line):
        """Parse baris — IMU CSV atau prediksi. Identik dengan SerialManager."""
        if self.on_raw_line:
            self.on_raw_line(line)

        if line.startswith("[PRED]"):
            parts = line.split()
            if len(parts) >= 4:
                gesture = parts[1]
                try:
                    confidence = float(parts[2])
                    latency = int(parts[3].replace("ms", ""))
                    if self.on_prediction:
                        self.on_prediction(gesture, confidence, latency)
                except (ValueError, IndexError):
                    pass
            return

        parts = line.split(",")
        if len(parts) == NUM_AXES:
            try:
                values = tuple(float(v) for v in parts)
                self._data_queue.put(values)
                if self.on_imu_data:
                    self.on_imu_data(*values)
            except ValueError:
                pass
