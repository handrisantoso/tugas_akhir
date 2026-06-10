# Perbandingan Model Float32 vs INT8 (LOSO — augmented)

Dataset: `dataset/gesture_dataset_augmented.csv` | 6-fold LOSO

| Model | Format | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| MLP | Float32 | 100.00% ± 0.00% | 100.00% ± 0.00% | 100.00% ± 0.00% | 1.0000 ± 0.0000 |
| MLP | INT8 | 100.00% ± 0.00% | 100.00% ± 0.00% | 100.00% ± 0.00% | 1.0000 ± 0.0000 |
| CNN1D | Float32 | 99.78% ± 0.33% | 99.78% ± 0.33% | 99.78% ± 0.33% | 0.9978 ± 0.0033 |
| CNN1D | INT8 | 99.84% ± 0.24% | 99.85% ± 0.23% | 99.84% ± 0.24% | 0.9984 ± 0.0024 |
| RF | Float32 | 100.00% ± 0.00% | 100.00% ± 0.00% | 100.00% ± 0.00% | 1.0000 ± 0.0000 |