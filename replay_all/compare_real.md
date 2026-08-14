# Perbandingan Model Float32 vs INT8 (LOSO — real)

Dataset: `C:\Users\sibga\OneDrive\Documents\gesture_glove\dataset\gesture_dataset.csv` | 7-fold LOSO

| Model | Format | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| MLP | Float32 | 92.54% ± 8.64% | 89.81% ± 13.29% | 92.54% ± 8.64% | 0.9058 ± 0.1161 |
| MLP | INT8 | 92.77% ± 8.66% | 89.90% ± 13.39% | 92.77% ± 8.66% | 0.9081 ± 0.1169 |
| CNN1D | Float32 | 85.75% ± 14.27% | 83.73% ± 19.24% | 85.75% ± 14.27% | 0.8280 ± 0.1836 |
| CNN1D | INT8 | 85.40% ± 14.12% | 83.56% ± 19.08% | 85.40% ± 14.12% | 0.8247 ± 0.1815 |
| RF | Float32 | 93.55% ± 8.83% | 90.76% ± 13.35% | 93.55% ± 8.83% | 0.9165 ± 0.1182 |