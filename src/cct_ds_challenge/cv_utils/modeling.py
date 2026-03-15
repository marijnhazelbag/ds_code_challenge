from __future__ import annotations
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights

from cv_utils.common import EarlyStopping

# ============================================================
# MODEL BUILDING
# ============================================================


def build_resnet18_binary(
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Build a binary ResNet18 classifier."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
    return model


# ============================================================
# TRAINING / EVALUATION
# ============================================================


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute binary classification metrics from probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["roc_auc"] = float("nan")
    return metrics



def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """Train the model for one epoch and return summary metrics."""
    model.train()
    running_loss = 0.0
    all_probs, all_targets = [], []

    for images, labels, _paths in data_loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        all_probs.extend(probs.tolist())
        all_targets.extend(labels.detach().cpu().numpy().ravel().tolist())

    epoch_loss = running_loss / len(data_loader.dataset)
    metrics = compute_metrics(np.array(all_targets), np.array(all_probs), threshold=0.5)
    metrics["loss"] = epoch_loss
    return metrics


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, threshold: float = 0.5):
    """Evaluate the model and return metrics, probabilities, and paths."""
    model.eval()
    running_loss = 0.0
    all_probs, all_targets, all_paths = [], [], []

    for images, labels, paths in data_loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        all_probs.extend(probs.tolist())
        all_targets.extend(labels.cpu().numpy().ravel().tolist())
        all_paths.extend(list(paths))

    epoch_loss = running_loss / len(data_loader.dataset)
    metrics = compute_metrics(np.array(all_targets), np.array(all_probs), threshold=threshold)
    metrics["loss"] = epoch_loss
    metrics["y_true"] = np.array(all_targets)
    metrics["y_prob"] = np.array(all_probs)
    metrics["paths"] = all_paths
    return metrics



def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    early_stopping_patience: int,
    min_delta: float,
    stage_label: str,
) -> Tuple[nn.Module, pd.DataFrame, Dict]:
    """Fit one model using validation F1 for checkpointing and early stopping."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    early_stopper = EarlyStopping(patience=early_stopping_patience, min_delta=min_delta)
    model.to(device)

    history_rows = []
    best_state = None
    best_score = -np.inf
    best_epoch = None

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, threshold=0.5)
        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history_rows.append(history_row)
        scheduler.step(val_metrics["f1"])

        print(
            f"[{stage_label}] Epoch {epoch:02d} | train_loss={history_row['train_loss']:.4f} | val_loss={history_row['val_loss']:.4f} | "
            f"val_f1={history_row['val_f1']:.4f} | val_acc={history_row['val_accuracy']:.4f}"
        )

        if history_row["val_f1"] > best_score + min_delta:
            best_score = history_row["val_f1"]
            best_epoch = epoch
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

        if early_stopper.step(history_row["val_f1"]):
            print(f"[{stage_label}] Early stopping triggered at epoch {epoch:02d}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history_rows), {"best_val_f1": best_score, "best_epoch": best_epoch}


# ============================================================
# THRESHOLD TUNING
# ============================================================


# ============================================================
# THRESHOLD TUNING
# ============================================================


def select_decision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Sequence[float],
    selection_metric: str = "f1",
) -> Tuple[float, pd.DataFrame]:
    """Select a classification threshold using validation-set probabilities only.

    The expected workflow is:
    1. Generate predicted probabilities on the validation set.
    2. Evaluate a grid of candidate thresholds on that validation data only.
    3. Select the best threshold using the requested selection metric.
    4. Apply the chosen threshold once to the untouched hold-out test set.

    This keeps threshold selection separate from final test evaluation.
    """
    valid_metrics = {"accuracy", "precision", "recall", "f1"}
    if selection_metric not in valid_metrics:
        raise ValueError(
            f"selection_metric must be one of {sorted(valid_metrics)}. "
            f"Got: {selection_metric}"
        )

    threshold_rows = []
    for threshold in thresholds:
        threshold_metrics = compute_metrics(y_true, y_prob, threshold=float(threshold))
        threshold_rows.append({"threshold": float(threshold), **threshold_metrics})

    threshold_df = pd.DataFrame(threshold_rows)

    # Preserve current default behavior:
    # rank by F1 first, then recall and precision as tie-breakers.
    if selection_metric == "f1":
        sort_columns = ["f1", "recall", "precision"]
    elif selection_metric == "recall":
        sort_columns = ["recall", "precision", "f1"]
    elif selection_metric == "precision":
        sort_columns = ["precision", "recall", "f1"]
    else:  # accuracy
        sort_columns = ["accuracy", "f1", "recall", "precision"]

    threshold_df = threshold_df.sort_values(sort_columns, ascending=False).reset_index(drop=True)
    best_threshold = float(threshold_df.iloc[0]["threshold"])
    return best_threshold, threshold_df
