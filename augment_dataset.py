"""
Create synthetic recording sessions from existing gesture data.

Each synthetic session applies a consistent per-session transformation
(time-stretch + amplitude scale + noise) to simulate a different person's
recording style.  Writes dataset/gesture_dataset_augmented.csv with
6 subjects total (2 real + 4 synthetic).
"""
import csv
import os
import numpy as np
from recorder import GestureRecorder
from config import WINDOW_SIZE, NUM_AXES, GESTURE_NAMES

SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]
OUT_CSV = os.path.join("dataset", "gesture_dataset_augmented.csv")
HEADER = ["label", "sample_id", "timestamp_ms", "subject_id"] + SENSOR_COLS
SAMPLE_RATE_MS = 10  # 100 Hz → 10 ms per row


def time_stretch(window, factor):
    """Resample a (WINDOW_SIZE, NUM_AXES) window along the time axis.

    factor > 1: compress the gesture in time (person moves faster — we see
                 more of the motion arc within the window)
    factor < 1: expand the gesture in time (person moves slower — we see
                 a smaller portion of the arc)
    """
    W, A = window.shape
    src = np.linspace(0, (W - 1) / factor, W).clip(0, W - 1)
    out = np.empty_like(window, dtype=np.float32)
    x = np.arange(W)
    for a in range(A):
        out[:, a] = np.interp(src, x, window[:, a])
    return out


def build_synthetic_session(df_src, subject_id, rng, stretch, amp_mean, amp_std, noise_std):
    """Return rows (list of lists) for one synthetic session.

    Parameters
    ----------
    df_src    : DataFrame — rows from ONE real session
    subject_id: str label for the new session
    rng       : numpy default_rng instance
    stretch   : float — time-stretch factor (0.8 – 1.25)
    amp_mean  : (NUM_AXES,) — per-axis amplitude scale centre
    amp_std   : float — stddev for per-axis amplitude jitter
    noise_std : float — Gaussian sensor noise stddev
    """
    rows = []
    sample_counter = 0
    for sid, grp in df_src.groupby("sample_id", sort=False):
        if len(grp) < WINDOW_SIZE:
            continue
        window = grp[SENSOR_COLS].values[:WINDOW_SIZE].astype(np.float32)  # (250, 6)

        # Per-session consistent amplitude scale (same for all windows in session)
        amp_scale = rng.normal(amp_mean, amp_std).clip(0.60, 1.40)

        # Time stretch
        window = time_stretch(window, stretch)

        # Amplitude scale
        window = window * amp_scale[np.newaxis, :]

        # Sample-level noise
        window += rng.normal(0, noise_std, window.shape).astype(np.float32)

        label = grp["label"].iloc[0]
        new_sid = f"{label}_{subject_id}_{sample_counter}"

        for t in range(WINDOW_SIZE):
            row = [label, new_sid, t * SAMPLE_RATE_MS, subject_id] + window[t].tolist()
            rows.append(row)

        sample_counter += 1
    return rows


def main():
    print("Loading dataset …")
    df, counts = GestureRecorder.load_csv("dataset/gesture_dataset.csv")
    print(f"  Loaded {sum(counts.values())} windows, columns: {list(df.columns)}")

    # Extract sessions
    df["_session"] = df["sample_id"].str.extract(r"^[a-z_]+_(\d+)_\d+$")
    sessions = df["_session"].unique()
    print(f"  Real sessions: {sessions}")

    rng = np.random.default_rng(42)

    # --- Synthetic session specs ---------------------------------------------------
    # Each spec: (base_session_idx, subject_id, stretch, amp_mean vector, amp_std, noise_std)
    #
    # stretch > 1 → faster gesture style (compress time)
    # stretch < 1 → slower gesture style (expand time)
    # amp_mean    → per-axis scaling centre (accel axes tend to vary more than gyro)
    # amp_std     → per-axis jitter within session
    # noise_std   → sensor noise
    # -------------------------------------------------------------------------------
    specs = [
        # From session 0 (1778852097)
        dict(base=0, subject_id="synth_A",
             stretch=1.18,
             amp_mean=np.array([1.10, 0.92, 1.05, 1.12, 0.94, 1.06]),
             amp_std=0.04, noise_std=0.09),
        dict(base=0, subject_id="synth_B",
             stretch=0.82,
             amp_mean=np.array([0.88, 1.12, 0.95, 0.87, 1.10, 0.93]),
             amp_std=0.04, noise_std=0.13),
        # From session 1 (1780652263)
        dict(base=1, subject_id="synth_C",
             stretch=1.22,
             amp_mean=np.array([1.18, 1.08, 0.87, 1.14, 0.91, 1.11]),
             amp_std=0.05, noise_std=0.07),
        dict(base=1, subject_id="synth_D",
             stretch=0.88,
             amp_mean=np.array([0.84, 0.91, 1.14, 0.83, 1.12, 0.89]),
             amp_std=0.05, noise_std=0.11),
    ]

    os.makedirs("dataset", exist_ok=True)
    written = 0
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        # --- Write real sessions first ---
        for row in df.drop(columns=["_session"]).itertuples(index=False):
            writer.writerow(list(row))
            written += 1

        # --- Write synthetic sessions ---
        for spec in specs:
            base_session = sessions[spec["base"]]
            df_src = df[df["_session"] == base_session].copy()
            print(f"  Generating {spec['subject_id']} "
                  f"(stretch={spec['stretch']:.2f}, "
                  f"noise={spec['noise_std']:.2f}) "
                  f"from session {base_session} …")
            rows = build_synthetic_session(
                df_src,
                subject_id=spec["subject_id"],
                rng=rng,
                stretch=spec["stretch"],
                amp_mean=spec["amp_mean"],
                amp_std=spec["amp_std"],
                noise_std=spec["noise_std"],
            )
            writer.writerows(rows)
            windows = len(rows) // WINDOW_SIZE
            print(f"    -> {windows} windows written")
            written += len(rows)

    print(f"\nDone -- {written} rows -> {OUT_CSV}")

    # Quick verify
    print("\nVerifying …")
    df2, counts2 = GestureRecorder.load_csv(OUT_CSV)
    df2["_session"] = df2["subject_id"]
    print(f"  Total windows : {sum(counts2.values())}")
    print(f"  Windows/gesture: {dict(counts2)}")
    print(f"  Subjects: {list(df2['_session'].unique())}")


if __name__ == "__main__":
    main()
