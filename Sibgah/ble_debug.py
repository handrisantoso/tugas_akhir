"""
BLE debug tool - run this first to diagnose data issues.

It will:
  1. Scan and let you pick a device
  2. Print all services and characteristics
  3. Subscribe to NOTIFY_UUID and print raw bytes + decoded text for 15 seconds
"""

import asyncio
import sys
from bleak import BleakClient, BleakScanner

NOTIFY_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
DURATION_SEC = 15
packet_count = 0


async def main():
    print("Scanning for BLE devices (5s)...")
    devices = await BleakScanner.discover(timeout=5.0)
    named = [d for d in devices if d.name]

    if not named:
        print("No named BLE devices found. Is the device powered on and advertising?")
        sys.exit(1)

    print("\nAvailable BLE devices:")
    for i, d in enumerate(named):
        print(f"  [{i}] {d.name}  ({d.address})")

    choice = 0
    if len(named) > 1:
        choice = int(input("Select device number: "))

    address = named[choice].address
    print(f"\nConnecting to {named[choice].name} ({address})...")

    async with BleakClient(address) as client:
        print(f"Connected: {client.is_connected}")

        # --- dump all services and characteristics ---
        print("\n=== Services & Characteristics ===")
        for service in client.services:
            print(f"\nService: {service.uuid}  ({service.description})")
            for char in service.characteristics:
                props = ", ".join(char.properties)
                print(f"  Char: {char.uuid}  [{props}]  ({char.description})")

        # --- check if our UUID exists ---
        all_uuids = [str(c.uuid).upper() for s in client.services for c in s.characteristics]
        target = NOTIFY_UUID.upper()
        if target not in all_uuids:
            print(f"\n[WARNING] NOTIFY_UUID {NOTIFY_UUID} NOT found on this device!")
            print("  Check the UUID in your ESP32 code and update NOTIFY_UUID in both scripts.")
        else:
            print(f"\n[OK] NOTIFY_UUID {NOTIFY_UUID} found on device.")

        # --- subscribe and print raw data ---
        print(f"\n=== Listening for notifications for {DURATION_SEC}s ===")
        print("(Each packet will show raw bytes + decoded text + field count)\n")

        def handle(sender, data: bytearray):
            global packet_count
            packet_count += 1
            try:
                text = data.decode("utf-8")
                visible = repr(text)
            except UnicodeDecodeError:
                text = None
                visible = "(not UTF-8)"

            print(f"[#{packet_count:04d}] raw={data.hex()}  len={len(data)}")
            print(f"        decoded={visible}")

            if text:
                for line in text.strip().split("\n"):
                    line = line.strip()
                    parts = line.split(",")
                    print(f"        line={repr(line)}  fields={len(parts)} -> {parts}")
            print()

        try:
            await client.start_notify(NOTIFY_UUID, handle)
        except Exception as e:
            print(f"[ERROR] start_notify failed: {e}")
            print("  The characteristic may not support notifications, or the UUID is wrong.")
            return

        await asyncio.sleep(DURATION_SEC)
        await client.stop_notify(NOTIFY_UUID)

        print(f"=== Done. Received {packet_count} packets in {DURATION_SEC}s ===")
        if packet_count == 0:
            print("[!] No data received. Possible causes:")
            print("    - ESP32 is not transmitting (check firmware)")
            print("    - Wrong characteristic UUID")
            print("    - Notify not enabled on ESP32 side (check if it requires CCCD write)")


if __name__ == "__main__":
    asyncio.run(main())
