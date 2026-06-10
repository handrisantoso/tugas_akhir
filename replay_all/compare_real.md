# Perbandingan Model Float32 vs INT8 (LOSO — real)

Dataset: `dataset/gesture_dataset.csv` | 5-fold LOSO

| Model | Format | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| MLP | Float32 | 91.11% ± 7.47% | 89.82% ± 11.36% | 91.11% ± 7.47% | 0.8950 ± 0.0991 |
| MLP | INT8 | 90.41% ± 9.43% | 88.64% ± 12.37% | 90.41% ± 9.43% | 0.8802 ± 0.1233 |
| CNN1D | Float32 | 93.85% ± 8.08% | 92.13% ± 12.16% | 93.85% ± 8.08% | 0.9254 ± 0.1066 |
| CNN1D | INT8 | 94.01% ± 7.95% | 96.23% ± 4.30% | 94.01% ± 7.95% | 0.9274 ± 0.1048 |
| RF | Float32 | 95.66% ± 7.82% | 93.56% ± 12.04% | 95.66% ± 7.82% | 0.9429 ± 0.1057 |