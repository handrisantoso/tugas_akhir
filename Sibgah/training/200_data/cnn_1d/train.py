"""
Phase 2: 1D CNN gesture classifier (PyTorch).
Note: TensorFlow does not yet support Python 3.14; PyTorch is used instead.

Architecture: Conv1D(64) -> Conv1D(128) -> GlobalAvgPool -> Dense(64) -> Softmax
Outputs saved to cnn_1d/output/:
  - training_history.png        (loss + accuracy per epoch)
  - confusion_matrix.png        (test set)
  - confusion_matrix_val.png    (validation set)
  - classification_report.txt   (test set)
  - metrics_summary.txt
  - model.pt
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.preprocessing import (
    load_file_list, split_file_list, get_class_names,
    extract_windows, fit_scaler, apply_scaler, compute_class_weights,
)
from common.evaluation import (
    plot_confusion_matrix, save_classification_report,
    save_metrics_summary, plot_history,
)
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "cleaned_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 100
BATCH_SIZE = 32
LR = 3e-4
PATIENCE = 10


class CNN1D(nn.Module):
    def __init__(self, n_features: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features) -> conv expects (batch, features, time)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.mean(dim=2)  # global average pooling over time
        return self.head(x)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y).long(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds = []
    for X_batch, y_batch in loader:
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
        preds.append(logits.argmax(1).numpy())
    return total_loss / total, correct / total, np.concatenate(preds)


def main() -> None:
    print("=" * 50)
    print("  Phase 2: 1D CNN (PyTorch)")
    print("=" * 50)

    file_list = load_file_list(DATA_DIR)
    class_names = get_class_names(file_list)
    print(f"\nClasses: {class_names}")

    train_files, val_files, test_files = split_file_list(file_list)
    print(f"Files  — train: {len(train_files)}  val: {len(val_files)}  test: {len(test_files)}")

    print("\nExtracting windows...")
    X_train, y_train = extract_windows(train_files, class_names)
    X_val,   y_val   = extract_windows(val_files,   class_names)
    X_test,  y_test  = extract_windows(test_files,  class_names)
    print(f"Windows — train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")

    scaler = fit_scaler(X_train)
    X_train = apply_scaler(X_train, scaler)
    X_val   = apply_scaler(X_val,   scaler)
    X_test  = apply_scaler(X_test,  scaler)

    num_classes = len(class_names)
    cw_dict = compute_class_weights(y_train, num_classes)
    cw_tensor = torch.tensor([cw_dict[i] for i in range(num_classes)], dtype=torch.float32)

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   BATCH_SIZE, shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  BATCH_SIZE, shuffle=False)

    model = CNN1D(n_features=X_train.shape[2], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, min_lr=1e-6
    )

    print(f"\nModel: {model}")
    print(f"\nTraining for up to {EPOCHS} epochs (early stop patience={PATIENCE})...")

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, cw_tensor)
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d} | loss {tr_loss:.4f}  acc {tr_acc:.4f} | "
                f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)

    _, val_acc_final, y_val_pred = evaluate(model, val_loader, criterion)
    _, test_acc_final, y_test_pred = evaluate(model, test_loader, criterion)

    test_f1_macro    = f1_score(y_test, y_test_pred, average="macro")
    test_f1_weighted = f1_score(y_test, y_test_pred, average="weighted")

    print(f"\nValidation accuracy : {val_acc_final:.4f}")
    print(f"Test accuracy       : {test_acc_final:.4f}")
    print(f"Test macro F1       : {test_f1_macro:.4f}")
    print(f"Test weighted F1    : {test_f1_weighted:.4f}\n")

    print("Saving outputs...")

    # Plot training history (dict-based, mirrors Keras history.history)
    class _History:
        def __init__(self, h):
            self.history = h

    plot_history(_History(history), OUTPUT_DIR)
    plot_confusion_matrix(y_test, y_test_pred, class_names, OUTPUT_DIR)
    plot_confusion_matrix(
        y_val, y_val_pred, class_names, OUTPUT_DIR,
        filename="confusion_matrix_val.png",
        title="Confusion Matrix (Validation Set)",
    )
    save_classification_report(y_test, y_test_pred, class_names, OUTPUT_DIR)
    save_metrics_summary(
        {
            "model": "1D_CNN",
            "val_accuracy": f"{val_acc_final:.4f}",
            "test_accuracy": f"{test_acc_final:.4f}",
            "test_macro_f1": f"{test_f1_macro:.4f}",
            "test_weighted_f1": f"{test_f1_weighted:.4f}",
            "epochs_trained": len(history["loss"]),
            "train_windows": len(X_train),
            "val_windows": len(X_val),
            "test_windows": len(X_test),
        },
        OUTPUT_DIR,
    )
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
    import joblib
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
