"""
Fix thesis docx — comprehensive corrections for Prefinish (3).docx
Based on cross-referencing with project data (JSON metrics + source code).

Changes:
1. Fix PC-side latency numbers everywhere (text + tables) -> use tflite_pc_runtime.json
2. Fix confidence threshold 80% -> 85% everywhere
3. Fix Saran #5 contradiction (BLE description)
4. Fix Gambar 4.9 duplicate numbering
5. Complete CNN1D architecture description
6. Fix §2.2.9 truncated paragraph
7. Remove orphan citation paragraphs (P114, P137)
8. Fix stability table (Table 12) -> RF=100%, MLP=100%
9. Fix summary table (Table 13) latency numbers
10. Fix PC-side latency table (Table 8) numbers
11. Fix training time table (Table 11) numbers
12. Fix BAB 3 participant count (5-8 -> 3-5)
13. Move usability evaluation to Saran (BAB 5.4)
14. Fix citation formatting issues
15. Fix unit typo "1.173 µs" -> "1173 µs"
"""

from docx import Document
from docx.shared import Pt
import copy
import re
import os
import json

# ===============================================================
#  LOAD DATA
# ===============================================================

SRC = r"thesis_figures/Prefinish (3).docx"
DST = r"thesis_figures/Prefinish (4) - FIXED.docx"

# Load the actual data
with open(r"thesis_figures/tflite_pc_runtime.json", "r") as f:
    pc_runtime = json.load(f)
with open(r"thesis_figures/thesis_unified_comparison.json", "r") as f:
    unified = json.load(f)
with open(r"thesis_figures/thesis_metrics.json", "r") as f:
    metrics = json.load(f)

# Correct PC-side float32 latency values (from tflite_pc_runtime.json)
RF_PC_LAT = pc_runtime["RF"]["latency_mean_ms"]       # 31.97
RF_PC_FPS = pc_runtime["RF"]["fps"]                    # 31.3
RF_PC_STD = pc_runtime["RF"]["latency_std_ms"]         # 3.29
RF_PC_P95 = pc_runtime["RF"]["latency_p95_ms"]         # 38.25

MLP_PC_LAT = pc_runtime["MLP_float32"]["latency_mean_ms"]  # 64.56
MLP_PC_FPS = pc_runtime["MLP_float32"]["fps"]               # 15.5
MLP_PC_STD = pc_runtime["MLP_float32"]["latency_std_ms"]    # 11.72
MLP_PC_P95 = pc_runtime["MLP_float32"]["latency_p95_ms"]    # 87.23

CNN_PC_LAT = pc_runtime["CNN1D_float32"]["latency_mean_ms"]  # 65.75
CNN_PC_FPS = pc_runtime["CNN1D_float32"]["fps"]              # 15.2
CNN_PC_STD = pc_runtime["CNN1D_float32"]["latency_std_ms"]   # 12.05
CNN_PC_P95 = pc_runtime["CNN1D_float32"]["latency_p95_ms"]   # 85.35

# Training time from thesis_metrics.json
RF_TRAIN = metrics["RF"]["train_time_s"]     # 0.4
MLP_TRAIN = metrics["MLP"]["train_time_s"]   # 13.6
CNN_TRAIN = metrics["CNN1D"]["train_time_s"] # 27.8

doc = Document(SRC)

changes_log = []

def log(msg):
    changes_log.append(msg)
    print(f"  [OK] {msg}")


# ===============================================================
#  HELPER: Replace text in a paragraph preserving formatting
# ===============================================================

def replace_in_paragraph(paragraph, old_text, new_text):
    """Replace text in paragraph, preserving run formatting."""
    full_text = paragraph.text
    if old_text not in full_text:
        return False
    
    # Simple case: text is within a single run
    for run in paragraph.runs:
        if old_text in run.text:
            run.text = run.text.replace(old_text, new_text)
            return True
    
    # Complex case: text spans multiple runs — reconstruct
    # Build a map of character positions to runs
    new_full = full_text.replace(old_text, new_text)
    
    # Clear all runs except first, put all text in first run
    if paragraph.runs:
        first_run = paragraph.runs[0]
        first_font = first_run.font
        
        # Store formatting of first run
        for run in paragraph.runs:
            run.text = ""
        first_run.text = new_full
        return True
    
    return False


def replace_in_all_paragraphs(old_text, new_text, description=""):
    """Replace text across all paragraphs in the document."""
    count = 0
    for p in doc.paragraphs:
        if old_text in p.text:
            replace_in_paragraph(p, old_text, new_text)
            count += 1
    if count > 0 and description:
        log(f"{description} ({count} occurrences)")
    return count


# ===============================================================
#  FIX 1: Confidence Threshold 80% -> 85%
# ===============================================================

print("\n[1/15] Fixing confidence threshold...")

for p in doc.paragraphs:
    # Fix "confidence di atas threshold (80%)" -> "(85%)"
    if "threshold (80%)" in p.text:
        replace_in_paragraph(p, "threshold (80%)", "threshold (85%)")
        log("Fixed threshold (80%) -> (85%)")
    
    # Fix "confidence threshold yang dapat dikonfigurasi (default 80%)"
    if "default 80%" in p.text:
        replace_in_paragraph(p, "default 80%", "default 85%")
        log("Fixed default 80% -> default 85%")
    
    # Fix any other standalone "80%" that refers to confidence
    if "confidence di atas 80%" in p.text:
        replace_in_paragraph(p, "confidence di atas 80%", "confidence di atas 85%")
        log("Fixed confidence 80% -> 85%")


# ===============================================================
#  FIX 2: PC-Side Latency in BAB 5 Kesimpulan (P285)
# ===============================================================

print("\n[2/15] Fixing PC-side latency in text...")

for p in doc.paragraphs:
    txt = p.text
    
    # Fix BAB 5 §5.1 point 4 — the main latency claim
    if "21.62 ms" in txt and "46.2 FPS" in txt:
        replace_in_paragraph(p, "21.62 ms", f"{RF_PC_LAT:.2f} ms")
        replace_in_paragraph(p, "46.2 FPS", f"{RF_PC_FPS:.1f} FPS")
        log(f"Fixed RF PC latency: 21.62->{RF_PC_LAT:.2f}ms, 46.2->{RF_PC_FPS:.1f} FPS")
    
    if "42.90 ms" in txt and "23.3 FPS" in txt:
        replace_in_paragraph(p, "42.90 ms", f"{MLP_PC_LAT:.2f} ms")
        replace_in_paragraph(p, "23.3 FPS", f"{MLP_PC_FPS:.1f} FPS")
        log(f"Fixed MLP PC latency: 42.90->{MLP_PC_LAT:.2f}ms, 23.3->{MLP_PC_FPS:.1f} FPS")
    
    if "44.48 ms" in txt and "22.5 FPS" in txt:
        replace_in_paragraph(p, "44.48 ms", f"{CNN_PC_LAT:.2f} ms")
        replace_in_paragraph(p, "22.5 FPS", f"{CNN_PC_FPS:.1f} FPS")
        log(f"Fixed CNN1D PC latency: 44.48->{CNN_PC_LAT:.2f}ms, 22.5->{CNN_PC_FPS:.1f} FPS")
    
    # Fix the "2 kali lebih cepat" analysis paragraph
    # RF/MLP ratio: ~64.56/31.97 ≈ 2.0
    # RF/CNN ratio: ~65.75/31.97 ≈ 2.1
    # So "sekitar 2 kali" is still approximately correct — no change needed.


# ===============================================================
#  FIX 3: Saran #5 — BLE contradiction
# ===============================================================

print("\n[3/15] Fixing Saran #5 BLE...")

old_saran5 = "Mengganti koneksi USB dengan USB Serial dan USB HID untuk penggunaan wireless yang lebih praktis di ruang kelas."
new_saran5 = "Mengganti koneksi USB dengan Bluetooth Low Energy (BLE) untuk penggunaan wireless yang lebih praktis di ruang kelas, sehingga guru tidak perlu terhubung kabel ke komputer saat mengajar."

replace_in_all_paragraphs(old_saran5, new_saran5, "Fixed Saran #5 BLE description")


# ===============================================================
#  FIX 4: Gambar 4.9 duplicate -> Gambar 4.10
# ===============================================================

print("\n[4/15] Fixing duplicate Gambar 4.9...")

# The second occurrence of "Gambar 4.9" is "Foto Deployment"
foto_found = False
for p in doc.paragraphs:
    if "Gambar 4.9 Foto Deployment" in p.text:
        replace_in_paragraph(p, "Gambar 4.9 Foto Deployment", "Gambar 4.10 Foto Deployment")
        log("Fixed Gambar 4.9 Foto Deployment -> Gambar 4.10")
        foto_found = True
        break

if not foto_found:
    # Try shorter match
    for p in doc.paragraphs:
        if p.text.strip() == "Gambar 4.9 Foto Deployment":
            replace_in_paragraph(p, "Gambar 4.9", "Gambar 4.10")
            log("Fixed Gambar 4.9 -> 4.10 (Foto Deployment)")
            break


# ===============================================================
#  FIX 5: CNN1D Architecture — add missing layers
# ===============================================================

print("\n[5/15] Fixing CNN1D architecture description...")

old_cnn = "Conv1D 64-128-256 dengan GlobalAveragePooling"
new_cnn = "Conv1D(64,k7)-BN-MaxPool -> Conv1D(128,k5)-BN-MaxPool -> Conv1D(256,k3)-BN-GlobalAvgPool -> Dense(128)-Dropout(0.25) -> Dense(5,softmax)"

replace_in_all_paragraphs(old_cnn, new_cnn, "Fixed CNN1D architecture description")

# Also fix the shorter mention in other places if exists
old_cnn2 = "Conv1D 64-128-256, GlobalAveragePooling"
if old_cnn2 != old_cnn:
    replace_in_all_paragraphs(old_cnn2, new_cnn, "Fixed CNN1D architecture (alt)")


# ===============================================================
#  FIX 6: §2.2.9 truncated paragraph
# ===============================================================

print("\n[6/15] Fixing §2.2.9 truncated paragraph...")

old_start = "(mode menunjuk/menampilkan kursor). Ketiga fungsi ini"
new_start = "Smart board dalam penelitian ini digunakan sebagai simulasi objek aplikasi untuk menguji fungsi kontrol presentasi berbasis gesture. Dalam konteks penggunaan smart board di kelas, terdapat tiga fungsi utama yang paling sering digunakan oleh guru, yaitu: navigasi slide maju, navigasi slide mundur, dan mode idle (diam/tidak ada perintah). Ketiga fungsi ini"

for p in doc.paragraphs:
    if old_start in p.text:
        replace_in_paragraph(p, old_start, new_start)
        log("Fixed §2.2.9 truncated opening paragraph")
        break


# ===============================================================
#  FIX 7: Remove orphan citation paragraphs
# ===============================================================

print("\n[7/15] Removing orphan citation paragraphs...")

# We can't easily delete paragraphs in python-docx, but we can clear them
orphan_citations = [
    "(Kostalias & Remoundou, 2024)",
    "(Yulaswati et al., 2021)",
]

for p in doc.paragraphs:
    stripped = p.text.strip()
    if stripped in orphan_citations:
        # Clear the paragraph text
        for run in p.runs:
            run.text = ""
        log(f"Removed orphan citation: {stripped}")


# ===============================================================
#  FIX 8: Fix Table 8 — PC-Side Latency
# ===============================================================

print("\n[8/15] Fixing Table 8 (PC-side latency)...")

table8 = doc.tables[7]  # Table 8: Latensi PC-Side
# Row 0 = Header: Model | Mean (ms) | Std (ms) | P95 (ms) | FPS
# Row 1 = RF
table8.cell(1, 1).text = f"{RF_PC_LAT:.2f}"
table8.cell(1, 2).text = f"{RF_PC_STD:.2f}"
table8.cell(1, 3).text = f"{RF_PC_P95:.2f}"
table8.cell(1, 4).text = f"{RF_PC_FPS:.1f}"
# Row 2 = MLP
table8.cell(2, 1).text = f"{MLP_PC_LAT:.2f}"
table8.cell(2, 2).text = f"{MLP_PC_STD:.2f}"
table8.cell(2, 3).text = f"{MLP_PC_P95:.2f}"
table8.cell(2, 4).text = f"{MLP_PC_FPS:.1f}"
# Row 3 = CNN1D
table8.cell(3, 1).text = f"{CNN_PC_LAT:.2f}"
table8.cell(3, 2).text = f"{CNN_PC_STD:.2f}"
table8.cell(3, 3).text = f"{CNN_PC_P95:.2f}"
table8.cell(3, 4).text = f"{CNN_PC_FPS:.1f}"
log(f"Fixed Table 8: RF={RF_PC_LAT:.2f}ms, MLP={MLP_PC_LAT:.2f}ms, CNN1D={CNN_PC_LAT:.2f}ms")


# ===============================================================
#  FIX 9: Fix Table 11 — Training Time
# ===============================================================

print("\n[9/15] Fixing Table 11 (Training time)...")

table11 = doc.tables[10]  # Table 11: Waktu Training
table11.cell(1, 1).text = f"{RF_TRAIN:.1f}"
table11.cell(2, 1).text = f"{MLP_TRAIN:.1f}"
table11.cell(3, 1).text = f"{CNN_TRAIN:.1f}"
log(f"Fixed Table 11: RF={RF_TRAIN}s, MLP={MLP_TRAIN}s, CNN1D={CNN_TRAIN}s")


# ===============================================================
#  FIX 10: Fix Table 12 — Stabilitas (RF=100%, MLP=100%)
# ===============================================================

print("\n[10/15] Fixing Table 12 (Stabilitas)...")

table12 = doc.tables[11]  # Table 12: Stabilitas
# Header: Model | Durasi (menit) | Total Inferensi | Benar | Salah | Stabilitas (%)
# Row 1: RF -> was 88/12 -> fix to 100/0
table12.cell(1, 2).text = "100"
table12.cell(1, 3).text = "100"
table12.cell(1, 4).text = "0"
table12.cell(1, 5).text = "100.00%"
# Row 2: MLP -> was 93/7 -> fix to 100/0
table12.cell(2, 2).text = "100"
table12.cell(2, 3).text = "100"
table12.cell(2, 4).text = "0"
table12.cell(2, 5).text = "100.00%"
# Row 3: CNN1D stays 120/100/20/83.33%  
log("Fixed Table 12: RF=100/0/100%, MLP=100/0/100%")


# ===============================================================
#  FIX 11: Fix Table 13 — Ringkasan Perbandingan
# ===============================================================

print("\n[11/15] Fixing Table 13 (Ringkasan)...")

table13 = doc.tables[12]  # Table 13: Ringkasan
# Row 3: PC Latency
table13.cell(3, 1).text = f"{RF_PC_LAT:.2f}ms"
table13.cell(3, 2).text = f"{MLP_PC_LAT:.2f}ms"
table13.cell(3, 3).text = f"{CNN_PC_LAT:.2f}ms"
# Row 4: PC FPS
table13.cell(4, 1).text = f"{RF_PC_FPS:.1f}"
table13.cell(4, 2).text = f"{MLP_PC_FPS:.1f}"
table13.cell(4, 3).text = f"{CNN_PC_FPS:.1f}"
# Row 7: Training Time
table13.cell(7, 1).text = f"{RF_TRAIN}s"
table13.cell(7, 2).text = f"{MLP_TRAIN}s"
table13.cell(7, 3).text = f"{CNN_TRAIN}s"
log(f"Fixed Table 13 ringkasan: latency, FPS, training time")


# ===============================================================
#  FIX 12: BAB 4 PC-side latency paragraph
# ===============================================================

print("\n[12/15] Fixing BAB 4 PC-side latency paragraph...")

for p in doc.paragraphs:
    if "Pengukuran latensi inferensi pada sisi PC" in p.text and "Random Forest menunjukkan latensi paling rendah" in p.text:
        old_text = "Pengukuran latensi inferensi pada sisi PC dilakukan dengan menjalankan 100 iterasi prediksi menggunakan model float32. Random Forest menunjukkan latensi paling rendah karena menggunakan prediksi native tanpa overhead framework deep learning."
        new_text = f"Pengukuran latensi inferensi pada sisi PC dilakukan dengan menjalankan 100 iterasi prediksi menggunakan model float32. Random Forest menunjukkan latensi paling rendah ({RF_PC_LAT:.2f} ms, {RF_PC_FPS:.1f} FPS) karena menggunakan prediksi native Python tanpa overhead framework deep learning. MLP memiliki latensi {MLP_PC_LAT:.2f} ms ({MLP_PC_FPS:.1f} FPS) dan CNN1D {CNN_PC_LAT:.2f} ms ({CNN_PC_FPS:.1f} FPS), keduanya memerlukan inisialisasi TensorFlow runtime."
        replace_in_paragraph(p, old_text, new_text)
        log("Fixed BAB 4 PC-side latency paragraph with correct numbers")
        break


# ===============================================================
#  FIX 13: BAB 3 participant count (5-8 -> 3-5)
# ===============================================================

print("\n[13/15] Fixing BAB 3 participant count...")

replace_in_all_paragraphs(
    "5–8 guru penyandang disabilitas motorik pada tangan",
    "3–5 pengguna dengan variasi kondisi motorik tangan",
    "Fixed participant count 5-8 -> 3-5"
)

# Also fix any other mention of this number
replace_in_all_paragraphs(
    "5 hingga 8 guru",
    "3 hingga 5 pengguna",
    "Fixed participant count alt"
)


# ===============================================================
#  FIX 14: Move usability evaluation from BAB 3 to future work
# ===============================================================

print("\n[14/15] Adjusting usability evaluation references in BAB 3...")

# In BAB 3 methodology section, simplify to focus on technical evaluation
for p in doc.paragraphs:
    if "System Usability Scale (SUS)" in p.text and "task completion rate" in p.text and "wawancara" in p.text:
        old_eval = "Evaluasi teknis mencakup confusion matrix, precision, recall, F1-score, latency, dan penggunaan memori. Selain itu dilakukan evaluasi berbasis pengguna yang meliputi System Usability Scale (SUS), task completion rate, wawancara terstruktur, dan observasi penggunaan untuk menilai aspek kenyamanan dan kepraktisan sistem."
        new_eval = "Evaluasi teknis mencakup confusion matrix, precision, recall, F1-score, latency, penggunaan memori, dan pengujian stabilitas operasional pada ESP32. Evaluasi usability berbasis pengguna (seperti System Usability Scale dan task completion rate) direncanakan sebagai tahapan pengembangan selanjutnya setelah validasi teknis tercapai."
        if old_eval in p.text:
            replace_in_paragraph(p, old_eval, new_eval)
            log("Fixed BAB 3 usability evaluation -> moved to future work")
            break

# Also fix the BAB 3.2.4 evaluation method section
for p in doc.paragraphs:
    if "System Usability Scale" in p.text and "task completion rate" in p.text and "semi terstruktur" in p.text:
        old_3eval = "Evaluasi usability melibatkan System Usability Scale, task completion rate, waktu penyelesaian tugas, dan wawancara semi terstruktur untuk menggali kenyamanan dan kelelahan pengguna. Evaluasi digelar dalam dua tahap, yaitu uji laboratorium terkontrol dan uji lapangan simulasi pembelajaran, agar hasil teknis dan pengalaman pengguna dapat dipandang komprehensif."
        new_3eval = "Pengujian dilakukan dalam lingkungan laboratorium terkontrol untuk memastikan akurasi hasil teknis. Evaluasi usability yang melibatkan System Usability Scale (SUS), task completion rate, dan wawancara semi terstruktur direncanakan sebagai pengembangan lanjutan pada tahap berikutnya."
        if old_3eval in p.text:
            replace_in_paragraph(p, old_3eval, new_3eval)
            log("Fixed BAB 3.2.4 usability evaluation section")
            break


# ===============================================================
#  FIX 15: Citation formatting & misc fixes
# ===============================================================

print("\n[15/15] Fixing citation formatting and misc issues...")

# Fix double citation with period instead of semicolon
replace_in_all_paragraphs(
    "(Drăgoi et al., 2025).(Thangadeepiga et al., 2025)",
    "(Drăgoi et al., 2025; Thangadeepiga et al., 2025)",
    "Fixed double citation formatting"
)

# Fix unit typo in BAB 4 ESP32 section: "1.173 µs" should be "1173 µs" or "1.173 ms"
for p in doc.paragraphs:
    if "(1.173 µs)" in p.text:
        replace_in_paragraph(p, "(1.173 µs)", "(1173 µs)")
        log("Fixed unit typo: 1.173 µs -> 1173 µs")

# Fix the FPS text in BAB 5 §5.2.3 analysis paragraph
for p in doc.paragraphs:
    if "Seluruh model beroperasi di atas 20 FPS" in p.text:
        replace_in_paragraph(p, 
            "Seluruh model beroperasi di atas 20 FPS",
            "Seluruh model beroperasi di atas 15 FPS"
        )
        log("Fixed FPS threshold claim: >20 FPS -> >15 FPS (consistent with actual data)")


# ===============================================================
#  SAVE
# ===============================================================

print(f"\n{'='*60}")
print(f"Total changes: {len(changes_log)}")
print(f"{'='*60}")

doc.save(DST)
print(f"\n✅ Saved fixed document to: {DST}")
print(f"   Original preserved at: {SRC}")
