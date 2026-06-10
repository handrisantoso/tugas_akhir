"""
split_subjects.py
=================
Memisahkan setiap subject (yang berisi data 2 orang) menjadi 2 sub-subject.

Asumsi: orang A merekam 75 sampel LEBIH DULU, lalu orang B merekam 75 sampel.
Split dilakukan per gesture berdasarkan urutan sample index (ascending).

Input : dataset/gesture_dataset.csv          (3 subjects, ~150 sampel/gesture)
Output: dataset/gesture_dataset_6subj.csv    (6 sub-subjects, ~75 sampel/gesture)

Sub-subject ID baru: {original_ID}_A  dan  {original_ID}_B

Usage:
    python split_subjects.py
    python split_subjects.py --verify     # hanya tampilkan statistik, tidak simpan
"""

import csv, re, os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INPUT_CSV  = os.path.join("dataset", "gesture_dataset.csv")
OUTPUT_CSV = os.path.join("dataset", "gesture_dataset_6subj.csv")

COLS_10 = ['label', 'sample_id', 'timestamp_ms', 'subject_id',
           'ax', 'ay', 'az', 'gx', 'gy', 'gz']


def load_raw(filepath):
    rows = []
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for fields in reader:
            if len(fields) == 10:
                rows.append(dict(zip(COLS_10, fields)))
            elif len(fields) == 9:
                m = re.match(r'^[a-z_]+_(\d+)_\d+$', fields[1])
                subj_id = m.group(1) if m else 'unknown'
                rows.append({
                    'label': fields[0], 'sample_id': fields[1],
                    'timestamp_ms': fields[2], 'subject_id': subj_id,
                    'ax': fields[3], 'ay': fields[4], 'az': fields[5],
                    'gx': fields[6], 'gy': fields[7], 'gz': fields[8],
                })
    return rows


def sample_index(sample_id):
    return int(sample_id.split('_')[-1])


def split_and_assign(rows):
    """Return new rows with subject_id replaced by {orig}_A or {orig}_B."""
    from collections import defaultdict

    # Group rows by (subject_id, label, sample_id) to find unique samples
    # Structure: {(subject_id, label): {sample_id: [rows...]}}
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r['subject_id'], r['label'])][r['sample_id']].append(r)

    new_rows = []
    stats = {}

    for (sid, label), sample_map in groups.items():
        # Sort unique sample_ids by their numeric index (= recording order)
        ordered = sorted(sample_map.keys(), key=sample_index)
        n = len(ordered)
        half = n // 2   # orang A: [:half], orang B: [half:]

        ids_A = set(ordered[:half])
        ids_B = set(ordered[half:])

        for sid_sample, sample_rows in sample_map.items():
            new_sid = f"{sid}_A" if sid_sample in ids_A else f"{sid}_B"
            for r in sample_rows:
                nr = dict(r)
                nr['subject_id'] = new_sid
                new_rows.append(nr)

        # Stats for verification
        ax_A = np.mean([float(r['ax']) for sid_s, rs in sample_map.items()
                        if sid_s in ids_A for r in rs])
        ax_B = np.mean([float(r['ax']) for sid_s, rs in sample_map.items()
                        if sid_s in ids_B for r in rs])
        stats[(sid, label)] = (half, n - half, ax_A, ax_B)

    return new_rows, stats


def print_stats(stats):
    print(f"\n{'Subject':<14} {'Gesture':<12} {'n_A':>5} {'n_B':>5} "
          f"{'ax_A':>8} {'ax_B':>8} {'diff':>6}  Note")
    print("-" * 75)
    gestures = ['idle', 'flick_up', 'wave_left', 'wave_right', 'flick_down']
    for sid in sorted(set(s for s, _ in stats)):
        for g in gestures:
            key = (sid, g)
            if key not in stats:
                continue
            n_A, n_B, ax_A, ax_B = stats[key]
            diff = abs(ax_A - ax_B)
            note = "BEDA SIGNIFIKAN" if diff > 2 else ("agak beda" if diff > 0.7 else "sama")
            print(f"  {sid[:12]:<12} {g:<12} {n_A:>5} {n_B:>5} "
                  f"{ax_A:>8.3f} {ax_B:>8.3f} {diff:>6.2f}  {note}")
        print()


def save_csv(rows, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLS_10)
        writer.writeheader()
        writer.writerows(rows)


def verify_output(filepath):
    """Quick sanity check on the saved CSV."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from recorder import GestureRecorder
    X, y, subj = GestureRecorder.load_dataset(filepath)
    unique = sorted(set(subj))
    print(f"\n=== Verifikasi {filepath} ===")
    print(f"  Total windows  : {len(y)}")
    print(f"  Jumlah subjects: {len(unique)}")
    print(f"  Subject IDs    : {unique}")
    print()
    gestures = ['idle', 'flick_up', 'wave_left', 'wave_right', 'flick_down']
    for s in unique:
        mask = subj == s
        counts = [int(np.sum(y[mask] == c)) for c in range(5)]
        print(f"  {s}: " + "  ".join(f"{gestures[i]}={counts[i]}" for i in range(5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="Hanya tampilkan statistik split, tidak simpan CSV")
    args = ap.parse_args()

    print(f"Input : {INPUT_CSV}")
    rows = load_raw(INPUT_CSV)
    print(f"  {len(rows)} baris dibaca")

    new_rows, stats = split_and_assign(rows)
    print_stats(stats)

    if args.verify:
        print("(--verify mode: tidak menyimpan file)")
        return

    save_csv(new_rows, OUTPUT_CSV)
    print(f"Saved : {OUTPUT_CSV}  ({len(new_rows)} baris)")

    verify_output(OUTPUT_CSV)

    print("\nSekarang jalankan LOSO dengan dataset baru:")
    print("  python compare_models_loso.py --csv dataset/gesture_dataset_6subj.csv")


if __name__ == "__main__":
    main()
