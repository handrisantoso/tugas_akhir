"""
Script untuk memisahkan gesture_dataset.csv menjadi subfolder per gesture,
dengan setiap rekaman (sample_id) disimpan sebagai file CSV terpisah.

Struktur output:
  dataset/
  ├── gesture_dataset.csv          (file asli, tidak diubah)
  ├── flick_down/
  │   ├── flick_down_XXXXX_0.csv
  │   ├── flick_down_XXXXX_1.csv
  │   └── ...
  ├── flick_up/
  │   ├── flick_up_XXXXX_0.csv
  │   └── ...
  ├── idle/
  │   ├── idle_XXXXX_0.csv
  │   └── ...
  ├── wave_left/
  │   ├── wave_left_XXXXX_0.csv
  │   └── ...
  └── wave_right/
      ├── wave_right_XXXXX_0.csv
      └── ...
"""

import csv
import os
from collections import defaultdict

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
INPUT_FILE = os.path.join(DATASET_DIR, "gesture_dataset.csv")

def main():
    # Baca semua data dan kelompokkan berdasarkan label -> sample_id
    data = defaultdict(lambda: defaultdict(list))
    
    print(f"Membaca file: {INPUT_FILE}")
    with open(INPUT_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # label, sample_id, timestamp_ms, ax, ay, az, gx, gy, gz
        
        for row in reader:
            label = row[0]
            sample_id = row[1]
            data[label][sample_id].append(row)
    
    # Header untuk file output (tanpa label dan sample_id karena sudah jadi nama folder/file)
    output_header = header[2:]  # timestamp_ms, ax, ay, az, gx, gy, gz
    
    total_files = 0
    
    for label in sorted(data.keys()):
        # Buat subfolder per gesture
        gesture_dir = os.path.join(DATASET_DIR, label)
        os.makedirs(gesture_dir, exist_ok=True)
        
        samples = data[label]
        print(f"\n[FOLDER] {label}/ - {len(samples)} rekaman")
        
        for sample_id in sorted(samples.keys()):
            rows = samples[sample_id]
            output_file = os.path.join(gesture_dir, f"{sample_id}.csv")
            
            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(output_header)
                for row in rows:
                    writer.writerow(row[2:])  # tulis data tanpa label & sample_id
            
            total_files += 1
        
        print(f"   [OK] {len(samples)} file CSV berhasil dibuat di {gesture_dir}")
    
    print(f"\n{'='*60}")
    print(f"[DONE] Selesai! Total {total_files} file rekaman dibuat.")
    print(f"   Lokasi: {DATASET_DIR}")

if __name__ == "__main__":
    main()
