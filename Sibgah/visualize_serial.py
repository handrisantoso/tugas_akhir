import sys
import asyncio
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from bleak import BleakClient, BleakScanner

NOTIFY_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
MAX_POINTS = 500
NUM_SENSORS = 2
VALUES_PER_SENSOR = 6
TOTAL_VALUES = NUM_SENSORS * VALUES_PER_SENSOR


def pick_device():
    print("Scanning for BLE devices...")
    devices = asyncio.run(BleakScanner.discover(timeout=5.0))
    named = [d for d in devices if d.name]
    if not named:
        print("No BLE devices found.")
        sys.exit(1)

    print("Available BLE devices:")
    for i, d in enumerate(named):
        print(f"  [{i}] {d.name} ({d.address})")

    if len(named) == 1:
        print(f"Auto-selecting {named[0].name}")
        return named[0].address

    choice = input("Select device number: ")
    return named[int(choice)].address


def main():
    address = pick_device()

    # 12 buffers: ax1 ay1 az1 gx1 gy1 gz1 ax2 ay2 az2 gx2 gy2 gz2
    buffers = [deque(maxlen=MAX_POINTS) for _ in range(TOTAL_VALUES)]

    accel_labels = ["ax", "ay", "az"]
    gyro_labels  = ["gx", "gy", "gz"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("MPU6050 Live Data (BLE)")
    titles = ["Sensor 1 - Accel", "Sensor 1 - Gyro", "Sensor 2 - Accel", "Sensor 2 - Gyro"]
    ax_list = list(axes.flat)
    all_lines = []

    for idx, ax in enumerate(ax_list):
        ax.set_title(titles[idx], fontsize=9)
        ax.set_xlim(0, MAX_POINTS)
        is_accel = idx % 2 == 0
        ax.set_ylim(-20000, 20000) if is_accel else ax.set_ylim(-35000, 35000)
        lbls = accel_labels if is_accel else gyro_labels
        group = [ax.plot([], [], label=lbl)[0] for lbl in lbls]
        ax.legend(loc="upper left", fontsize=7)
        all_lines.append(group)

    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    running = threading.Event()
    running.set()
    connected = threading.Event()

    def handle_data(_sender, data: bytearray):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != TOTAL_VALUES:
                continue
            try:
                values = [int(v) for v in parts]
            except ValueError:
                continue
            for buf, val in zip(buffers, values):
                buf.append(val)

    async def ble_task():
        try:
            async with BleakClient(address) as client:
                print(f"Connected to {address}")
                connected.set()
                await client.start_notify(NOTIFY_UUID, handle_data)
                while running.is_set():
                    await asyncio.sleep(0.1)
                await client.stop_notify(NOTIFY_UUID)
        except Exception as e:
            print(f"BLE error: {e}")
            connected.set()

    ble_thread = threading.Thread(target=lambda: asyncio.run(ble_task()), daemon=True)
    ble_thread.start()

    print("Waiting for BLE connection...")
    connected.wait(timeout=15)

    # buf layout per sensor: [accel(3), gyro(3)]
    buf_groups = []
    for s in range(NUM_SENSORS):
        offset = s * VALUES_PER_SENSOR
        buf_groups.append([offset, offset + 1, offset + 2])        # accel
        buf_groups.append([offset + 3, offset + 4, offset + 5])    # gyro

    def update(_frame):
        if not buffers[0]:
            return [line for group in all_lines for line in group]

        x = range(len(buffers[0]))
        for group_idx, indices in enumerate(buf_groups):
            for line_idx, buf_idx in enumerate(indices):
                all_lines[group_idx][line_idx].set_data(x, list(buffers[buf_idx]))
            ax_list[group_idx].set_xlim(0, max(len(buffers[0]), MAX_POINTS))

        return [line for group in all_lines for line in group]

    _anim = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()
    running.clear()


if __name__ == "__main__":
    main()
