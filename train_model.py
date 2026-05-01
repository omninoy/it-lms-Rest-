from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import MODEL_DIR, MODEL_PATH
from app.model import DigitCNN


def prepare_data(test_size: float = 0.2, random_state: int = 42):
    digits = load_digits()
    x = digits.images.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)

    x = np.stack(
        [
            np.kron(img, np.ones((4, 4), dtype=np.float32))
            for img in x
        ],
        axis=0,
    )
    x = np.expand_dims(x, axis=1)
    x = (x - 0.5) / 0.5

    x, y = shuffle(x, y, random_state=random_state)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    return train_ds, val_ds


def train(epochs: int = 12, lr: float = 1e-3, batch_size: int = 64):
    train_ds, val_ds = prepare_data()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cpu")
    model = DigitCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total if total else 0.0
        print(f"Epoch {epoch + 1}/{epochs} - val_accuracy={acc:.4f}")

    return model


def save_model(model: torch.nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")


if __name__ == "__main__":
    trained_model = train()
    save_model(trained_model, MODEL_PATH)
