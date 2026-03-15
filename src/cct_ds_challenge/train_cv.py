from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


# ============================================================
# Configuration
# ============================================================


@dataclass
class RunConfig:
    project_root: str = "."
    image_dir: str = "data/raw/swimming_pools"
    output_dir: str = "reports/exports/cv_baseline"
    dev_sample_per_class: Optional[int] = None
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    baseline_epochs: int = 5
    improved_epochs: int = 5
    learning_rate: float = 1e-3
    improved_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    val_size: float = 0.15
    test_size: float = 0.15
    decision_threshold_grid: Tuple[float, ...] = tuple(np.round(np.arange(0.20, 0.81, 0.05), 2))
    use_mlflow: bool = False
    mlflow_experiment: str = "cct_swimming_pool_cv"
    max_error_examples: int = 12
    report_title: str = "Swimming Pool Classification Report"
    early_stopping_patience: int = 2
    min_delta: float = 1e-4


# ============================================================
# Utilities
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Timer:
    def __init__(self) -> None:
        self._times: Dict[str, float] = {}
        self._start: Dict[str, float] = {}

    def start(self, name: str) -> None:
        self._start[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        if name not in self._start:
            raise KeyError(f"Timer '{name}' was not started")
        self._times[name] = time.perf_counter() - self._start[name]

    def summary_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"step": k, "seconds": v, "minutes": v / 60.0} for k, v in self._times.items()]
        ).sort_values("seconds", ascending=False)

    def to_dict(self) -> Dict[str, float]:
        return dict(self._times)


class OptionalMLflowLogger:
    def __init__(self, enabled: bool, experiment_name: str) -> None:
        self.enabled = enabled
        self.active = False
        self.mlflow = None
        if enabled:
            try:
                import mlflow

                self.mlflow = mlflow
                self.mlflow.set_experiment(experiment_name)
                self.active = True
            except Exception as exc:
                print(f"[mlflow] Disabled. Could not import/configure mlflow: {exc}")
                self.enabled = False
                self.active = False

    def start_run(self, run_name: str) -> None:
        if self.active:
            self.mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict) -> None:
        if self.active:
            self.mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self.active:
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                    self.mlflow.log_metric(key, float(value), step=step)

    def log_artifact(self, path: str) -> None:
        if self.active and os.path.exists(path):
            self.mlflow.log_artifact(path)

    def end_run(self) -> None:
        if self.active:
            self.mlflow.end_run()


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ============================================================
# Path handling and data checks
# ============================================================


def resolve_paths(cfg: RunConfig) -> Dict[str, Path]:
    root = Path(cfg.project_root).resolve()
    image_dir = (root / cfg.image_dir).resolve()
    output_dir = (root / cfg.output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    return {
        "root": root,
        "image_dir": image_dir,
        "output_dir": output_dir,
        "figures_dir": output_dir / "figures",
        "models_dir": output_dir / "models",
    }


ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}



def validate_dataset_structure(paths: Dict[str, Path]) -> None:
    image_dir = paths["image_dir"]
    yes_dir = image_dir / "yes"
    no_dir = image_dir / "no"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not yes_dir.exists():
        raise FileNotFoundError(f"Expected positive class directory: {yes_dir}")
    if not no_dir.exists():
        raise FileNotFoundError(f"Expected negative class directory: {no_dir}")



def _list_valid_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES])



def load_labels_dataframe(cfg: RunConfig) -> pd.DataFrame:
    paths = resolve_paths(cfg)
    validate_dataset_structure(paths)

    yes_dir = paths["image_dir"] / "yes"
    no_dir = paths["image_dir"] / "no"

    rows = []
    for folder, label in [(yes_dir, 1), (no_dir, 0)]:
        for image_path in _list_valid_images(folder):
            rows.append(
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "label": label,
                    "class_name": "yes" if label == 1 else "no",
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {paths['image_dir']} with supported suffixes {sorted(ALLOWED_SUFFIXES)}")

    class_counts = df["label"].value_counts().to_dict()
    if len(class_counts) < 2:
        raise RuntimeError(f"Expected two classes. Found counts: {class_counts}")

    bad_files = []
    for sample_path in df["image_path"].sample(min(20, len(df)), random_state=cfg.seed).tolist():
        try:
            with Image.open(sample_path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError) as exc:
            bad_files.append((sample_path, str(exc)))
    if bad_files:
        raise RuntimeError(f"Found unreadable images, e.g. {bad_files[:3]}")

    print(f"Loaded {len(df)} images from {paths['image_dir']}")
    print(f"Class counts: {df['label'].value_counts().sort_index().to_dict()}")
    return df



def make_balanced_subset(df: pd.DataFrame, sample_per_class: int, seed: int) -> pd.DataFrame:
    frames = []
    rng = np.random.default_rng(seed)
    for label, grp in df.groupby("label"):
        take_n = min(sample_per_class, len(grp))
        idx = rng.choice(grp.index.values, size=take_n, replace=False)
        frames.append(grp.loc[idx])
    subset = pd.concat(frames, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"Development subset class counts: {subset['label'].value_counts().to_dict()}")
    return subset



def stratified_train_val_test_split(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=df["label"],
        random_state=seed,
    )
    rel_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        stratify=temp_df["label"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


class PoolDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return image, label, row["image_path"]



def get_transforms(image_size: int, improved: bool = False):
    weights = ResNet18_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    if not improved:
        train_tfms = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        train_tfms = transforms.Compose(
            [
                transforms.Resize((image_size + 16, image_size + 16)),
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tfms, eval_tfms



def make_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    num_workers: int,
    improved: bool = False,
):
    train_tfms, eval_tfms = get_transforms(image_size=image_size, improved=improved)
    train_ds = PoolDataset(train_df, transform=train_tfms)
    val_ds = PoolDataset(val_df, transform=eval_tfms)
    test_ds = PoolDataset(test_df, transform=eval_tfms)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


# ============================================================
# Modelling
# ============================================================


def build_resnet18_binary(pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, 1))
    return model



def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_probs, all_targets = [], []

    for images, labels, _paths in loader:
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

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_targets), np.array(all_probs), threshold=0.5)
    metrics["loss"] = epoch_loss
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float = 0.5):
    model.eval()
    running_loss = 0.0
    all_probs, all_targets, all_paths = [], [], []

    for images, labels, paths in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        all_probs.extend(probs.tolist())
        all_targets.extend(labels.cpu().numpy().ravel().tolist())
        all_paths.extend(list(paths))

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_targets), np.array(all_probs), threshold=threshold)
    metrics["loss"] = epoch_loss
    metrics["y_true"] = np.array(all_targets)
    metrics["y_prob"] = np.array(all_probs)
    metrics["paths"] = all_paths
    return metrics



def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["roc_auc"] = float("nan")
    return out



def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Sequence[float]) -> Tuple[float, pd.DataFrame]:
    rows = []
    for thr in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=float(thr))
        rows.append({"threshold": float(thr), **m})
    df = pd.DataFrame(rows).sort_values(["f1", "recall", "precision"], ascending=False).reset_index(drop=True)
    best_thr = float(df.iloc[0]["threshold"])
    return best_thr, df



def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    mlflow_logger: OptionalMLflowLogger,
    run_name: str,
    early_stopping_patience: int,
    min_delta: float,
    stage_label: str,
) -> Tuple[nn.Module, pd.DataFrame, Dict]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    early_stopper = EarlyStopping(patience=early_stopping_patience, min_delta=min_delta)
    model.to(device)

    history = []
    best_state = None
    best_score = -np.inf
    best_epoch = None

    mlflow_logger.start_run(run_name)
    mlflow_logger.log_params(
        {
            "epochs": epochs,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "device": str(device),
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
            "early_stopping_patience": early_stopping_patience,
            "min_delta": min_delta,
            "stage": stage_label,
        }
    )

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, threshold=0.5)
        row = {
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
        history.append(row)
        scheduler.step(val_metrics["f1"])
        mlflow_logger.log_metrics(row, step=epoch)

        print(
            f"[{stage_label}] Epoch {epoch:02d} | train_loss={row['train_loss']:.4f} | val_loss={row['val_loss']:.4f} | "
            f"val_f1={row['val_f1']:.4f} | val_acc={row['val_accuracy']:.4f}"
        )

        if row["val_f1"] > best_score + min_delta:
            best_score = row["val_f1"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if early_stopper.step(row["val_f1"]):
            print(f"[{stage_label}] Early stopping triggered at epoch {epoch:02d}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    mlflow_logger.end_run()
    return model, pd.DataFrame(history), {"best_val_f1": best_score, "best_epoch": best_epoch}


# ============================================================
# Reporting helpers
# ============================================================


def save_training_curves(history_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_f1"], label="train_f1")
    axes[1].plot(history_df["epoch"], history_df["val_f1"], label="val_f1")
    axes[1].set_title("F1 score")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def save_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, output_path: Path, title: str) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No pool", "Pool"])
    ax.set_yticklabels(["No pool", "Pool"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def save_threshold_plot(threshold_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric in ["accuracy", "precision", "recall", "f1"]:
        ax.plot(threshold_df["threshold"], threshold_df[metric], label=metric)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Validation metrics across thresholds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def save_calibration_plot(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def save_error_panel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    paths: Sequence[str],
    threshold: float,
    output_path: Path,
    max_examples: int = 12,
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    false_pos_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    false_neg_idx = np.where((y_true == 1) & (y_pred == 0))[0]

    selected = list(false_pos_idx[: max_examples // 2]) + list(false_neg_idx[: max_examples // 2])
    if len(selected) == 0:
        print("No FP/FN examples to show.")
        return

    n = len(selected)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, idx in zip(axes, selected):
        image = Image.open(paths[idx]).convert("RGB")
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"true={int(y_true[idx])}, pred={int(y_pred[idx])}\np={y_prob[idx]:.2f}", fontsize=9)

    fig.suptitle("Error analysis: false positives and false negatives")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def metrics_to_frame(metrics: Dict[str, float], stage: str, threshold: float) -> pd.DataFrame:
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "loss"]
    rows = [{"stage": stage, "threshold": threshold, **{k: metrics.get(k) for k in keys}}]
    return pd.DataFrame(rows)



def dataset_summary_frame(df: pd.DataFrame, name: str) -> Dict[str, object]:
    counts = df["label"].value_counts().sort_index().to_dict()
    return {
        "split": name,
        "rows": len(df),
        "no_pool": counts.get(0, 0),
        "pool": counts.get(1, 0),
        "pool_rate": counts.get(1, 0) / len(df) if len(df) else np.nan,
    }



def save_html_report(
    cfg: RunConfig,
    class_summary: pd.DataFrame,
    timings_df: pd.DataFrame,
    baseline_metrics_df: pd.DataFrame,
    improved_metrics_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    paths: Dict[str, Path],
    best_threshold: float,
    run_notes: List[str],
) -> Path:
    report_path = paths["output_dir"] / "cv_report.html"

    def rel(p: Path) -> str:
        return os.path.relpath(p, report_path.parent)

    baseline_curve = paths["figures_dir"] / "baseline_training_curves.png"
    improved_curve = paths["figures_dir"] / "improved_training_curves.png"
    baseline_cm = paths["figures_dir"] / "baseline_confusion_matrix.png"
    improved_cm = paths["figures_dir"] / "improved_confusion_matrix.png"
    threshold_plot = paths["figures_dir"] / "threshold_tuning.png"
    calib_plot = paths["figures_dir"] / "improved_calibration.png"
    errors_plot = paths["figures_dir"] / "error_panel.png"

    notes_html = "".join(f"<li>{note}</li>" for note in run_notes)

    html = f"""
    <html>
    <head>
        <meta charset='utf-8'>
        <title>{cfg.report_title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.5; max-width: 1100px; }}
            h1, h2, h3 {{ color: #16324f; }}
            table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
            th {{ background: #f4f6f8; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin: 10px 0 20px 0; }}
            .note {{ background: #f8fbff; border-left: 4px solid #5b8def; padding: 12px; }}
            code {{ background: #f4f4f4; padding: 2px 4px; }}
        </style>
    </head>
    <body>
        <h1>{cfg.report_title}</h1>
        <p>This report presents a fair two-stage modelling workflow for swimming pool classification from imagery. Both the initial and improved solutions are trained and selected on the same train and validation splits, and both are compared on the same untouched hold-out test set. This makes the final comparison easier to interpret.</p>

        <div class='note'>
            <strong>Crux of the problem.</strong> This is not only a modelling problem. It is also a data quality and decision-threshold problem. Small visual structures, varied backgrounds, and ambiguous edge cases can all lead to false positives and false negatives. The modelling strategy therefore combines a lightweight architecture with explicit validation discipline, early stopping, and error inspection.
        </div>

        <h2>1. Execution summary</h2>
        <ul>{notes_html}</ul>

        <h2>2. Data summary</h2>
        {class_summary.to_html(index=False)}

        <h2>3. Initial solution</h2>
        <p><strong>Model:</strong> ResNet18 transfer learning<br>
        <strong>Preprocessing:</strong> resize to {cfg.image_size}×{cfg.image_size}, ImageNet normalization<br>
        <strong>Selection:</strong> best validation F1 with early stopping<br>
        <strong>Test evaluation:</strong> untouched hold-out test set</p>
        {baseline_metrics_df.to_html(index=False)}
        <img src='{rel(baseline_curve)}' alt='Baseline training curves'>
        <img src='{rel(baseline_cm)}' alt='Baseline confusion matrix'>

        <h2>4. Improved solution</h2>
        <p><strong>Changes from baseline:</strong></p>
        <ul>
            <li>Same train, validation, and hold-out test split as the baseline</li>
            <li>Moderate augmentation rather than aggressive image distortion</li>
            <li>More trainable parameters via fine-tuning</li>
            <li>Threshold tuning on validation probabilities</li>
            <li>Error analysis on false positives and false negatives</li>
            <li>Calibration check for prediction probabilities</li>
            <li>Early stopping based on validation F1</li>
        </ul>
        <p><strong>Selected threshold:</strong> {best_threshold:.2f}</p>
        {improved_metrics_df.to_html(index=False)}
        <img src='{rel(improved_curve)}' alt='Improved training curves'>
        <img src='{rel(threshold_plot)}' alt='Threshold tuning'>
        <img src='{rel(improved_cm)}' alt='Improved confusion matrix'>
        <img src='{rel(calib_plot)}' alt='Calibration plot'>
        <img src='{rel(errors_plot)}' alt='Error analysis panel'>

        <h2>5. Threshold tuning evidence</h2>
        {threshold_df.to_html(index=False)}

        <h2>6. Runtime and resource awareness</h2>
        {timings_df.to_html(index=False)}

        <h2>7. Interpretation</h2>
        <p>The initial solution answers a modest but important question: can a standard transfer-learning model separate pool from no-pool under a simple setup? The improved solution then focuses on reliability while keeping the comparison fair by using the same data split. Rather than changing data volume between stages, it improves the experimental design and the decision rule.</p>

        <h2>8. Reproducibility</h2>
        <p>All outputs are written under <code>{cfg.output_dir}</code>, including figures, saved model weights, metrics, timing logs, the saved configuration, and this HTML report. The script is intended to run end-to-end with a single command and no manual interaction.</p>
    </body>
    </html>
    """
    report_path.write_text(html, encoding="utf-8")
    return report_path


# ============================================================
# Pipeline
# ============================================================


def run_baseline_and_improved(cfg: RunConfig) -> Dict:
    set_seed(cfg.seed)
    timer = Timer()
    mlflow_logger = OptionalMLflowLogger(cfg.use_mlflow, cfg.mlflow_experiment)
    paths = resolve_paths(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timer.start("load_labels")
    full_df = load_labels_dataframe(cfg)
    timer.stop("load_labels")

    if cfg.dev_sample_per_class is not None:
        timer.start("dev_sample")
        modelling_df = make_balanced_subset(full_df, cfg.dev_sample_per_class, cfg.seed)
        timer.stop("dev_sample")
        data_scope = f"development_subset_{cfg.dev_sample_per_class}_per_class"
    else:
        modelling_df = full_df.copy()
        data_scope = "full_data"

    timer.start("split_data")
    train_df, val_df, test_df = stratified_train_val_test_split(
        modelling_df, val_size=cfg.val_size, test_size=cfg.test_size, seed=cfg.seed
    )
    timer.stop("split_data")

    class_summary = pd.DataFrame(
        [
            dataset_summary_frame(full_df, "full_available"),
            dataset_summary_frame(modelling_df, f"modelling_source_{data_scope}"),
            dataset_summary_frame(train_df, "train"),
            dataset_summary_frame(val_df, "validation"),
            dataset_summary_frame(test_df, "holdout_test"),
        ]
    )

    run_notes = [
        f"Device used: {device}",
        f"Data scope used for both baseline and improved models: {data_scope}",
        f"Fixed random seed: {cfg.seed}",
        f"Hold-out test set created once and used only for final comparison.",
        f"Early stopping patience: {cfg.early_stopping_patience}",
        f"Outputs written to: {paths['output_dir']}",
    ]

    criterion = nn.BCEWithLogitsLoss()

    timer.start("baseline_dataloaders")
    baseline_train_loader, baseline_val_loader, baseline_test_loader = make_dataloaders(
        train_df, val_df, test_df,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        improved=False,
    )
    timer.stop("baseline_dataloaders")

    timer.start("improved_dataloaders")
    improved_train_loader, improved_val_loader, improved_test_loader = make_dataloaders(
        train_df, val_df, test_df,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        improved=True,
    )
    timer.stop("improved_dataloaders")

    timer.start("baseline_training")
    baseline_model = build_resnet18_binary(pretrained=True, freeze_backbone=True)
    baseline_model, baseline_history, baseline_meta = fit_model(
        model=baseline_model,
        train_loader=baseline_train_loader,
        val_loader=baseline_val_loader,
        device=device,
        epochs=cfg.baseline_epochs,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        mlflow_logger=mlflow_logger,
        run_name="baseline_resnet18",
        early_stopping_patience=cfg.early_stopping_patience,
        min_delta=cfg.min_delta,
        stage_label="Baseline",
    )
    timer.stop("baseline_training")

    timer.start("baseline_evaluation")
    baseline_test_metrics = evaluate(baseline_model, baseline_test_loader, criterion, device, threshold=0.5)
    timer.stop("baseline_evaluation")

    torch.save(baseline_model.state_dict(), paths["models_dir"] / "baseline_resnet18.pt")
    baseline_history.to_csv(paths["output_dir"] / "baseline_history.csv", index=False)
    save_training_curves(baseline_history, paths["figures_dir"] / "baseline_training_curves.png", "Initial solution: baseline training curves")
    save_confusion_matrix(
        baseline_test_metrics["y_true"],
        baseline_test_metrics["y_prob"],
        threshold=0.5,
        output_path=paths["figures_dir"] / "baseline_confusion_matrix.png",
        title="Baseline confusion matrix (hold-out test, threshold=0.50)",
    )
    baseline_metrics_df = metrics_to_frame(baseline_test_metrics, stage="initial_solution", threshold=0.5)
    baseline_metrics_df.to_csv(paths["output_dir"] / "baseline_metrics.csv", index=False)

    timer.start("improved_training")
    improved_model = build_resnet18_binary(pretrained=True, freeze_backbone=False)
    improved_model, improved_history, improved_meta = fit_model(
        model=improved_model,
        train_loader=improved_train_loader,
        val_loader=improved_val_loader,
        device=device,
        epochs=cfg.improved_epochs,
        lr=cfg.improved_learning_rate,
        weight_decay=cfg.weight_decay,
        mlflow_logger=mlflow_logger,
        run_name="improved_resnet18",
        early_stopping_patience=cfg.early_stopping_patience,
        min_delta=cfg.min_delta,
        stage_label="Improved",
    )
    timer.stop("improved_training")

    timer.start("threshold_tuning")
    improved_val_metrics = evaluate(improved_model, improved_val_loader, criterion, device, threshold=0.5)
    best_threshold, threshold_df = tune_threshold(
        improved_val_metrics["y_true"],
        improved_val_metrics["y_prob"],
        cfg.decision_threshold_grid,
    )
    timer.stop("threshold_tuning")

    timer.start("improved_evaluation")
    improved_test_metrics = evaluate(improved_model, improved_test_loader, criterion, device, threshold=best_threshold)
    timer.stop("improved_evaluation")

    torch.save(improved_model.state_dict(), paths["models_dir"] / "improved_resnet18.pt")
    improved_history.to_csv(paths["output_dir"] / "improved_history.csv", index=False)
    threshold_df.to_csv(paths["output_dir"] / "threshold_tuning.csv", index=False)
    save_training_curves(improved_history, paths["figures_dir"] / "improved_training_curves.png", "Improved solution: training curves")
    save_threshold_plot(threshold_df, paths["figures_dir"] / "threshold_tuning.png")
    save_confusion_matrix(
        improved_test_metrics["y_true"],
        improved_test_metrics["y_prob"],
        threshold=best_threshold,
        output_path=paths["figures_dir"] / "improved_confusion_matrix.png",
        title=f"Improved confusion matrix (hold-out test, threshold={best_threshold:.2f})",
    )
    save_calibration_plot(improved_test_metrics["y_true"], improved_test_metrics["y_prob"], paths["figures_dir"] / "improved_calibration.png")
    save_error_panel(
        improved_test_metrics["y_true"],
        improved_test_metrics["y_prob"],
        improved_test_metrics["paths"],
        threshold=best_threshold,
        output_path=paths["figures_dir"] / "error_panel.png",
        max_examples=cfg.max_error_examples,
    )
    improved_metrics_df = metrics_to_frame(improved_test_metrics, stage="improved_solution", threshold=best_threshold)
    improved_metrics_df.to_csv(paths["output_dir"] / "improved_metrics.csv", index=False)

    timings_df = timer.summary_df()
    timings_df.to_csv(paths["output_dir"] / "timings.csv", index=False)
    class_summary.to_csv(paths["output_dir"] / "class_summary.csv", index=False)

    run_metadata = {
        "baseline_best_epoch": baseline_meta.get("best_epoch"),
        "baseline_best_val_f1": baseline_meta.get("best_val_f1"),
        "improved_best_epoch": improved_meta.get("best_epoch"),
        "improved_best_val_f1": improved_meta.get("best_val_f1"),
        "best_threshold_improved": best_threshold,
        "data_scope": data_scope,
    }
    (paths["output_dir"] / "model_selection_summary.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    config_path = paths["output_dir"] / "run_config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    report_path = save_html_report(
        cfg=cfg,
        class_summary=class_summary,
        timings_df=timings_df,
        baseline_metrics_df=baseline_metrics_df,
        improved_metrics_df=improved_metrics_df,
        threshold_df=threshold_df,
        paths=paths,
        best_threshold=best_threshold,
        run_notes=run_notes,
    )

    summary = {
        "device": str(device),
        "baseline_metrics": baseline_metrics_df.to_dict(orient="records"),
        "improved_metrics": improved_metrics_df.to_dict(orient="records"),
        "best_threshold": best_threshold,
        "report_path": str(report_path),
        "timings": timer.to_dict(),
        "output_dir": str(paths["output_dir"]),
        "config_path": str(config_path),
        "model_selection_summary": run_metadata,
    }
    with open(paths["output_dir"] / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Run complete.")
    print(json.dumps(summary, indent=2))
    print(f"HTML report written to: {report_path}")
    return summary


# ============================================================
# CLI
# ============================================================


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Train baseline and improved swimming pool classifier.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--image-dir", default="data/raw/swimming_pools")
    parser.add_argument("--output-dir", default="reports/exports/cv_baseline")
    parser.add_argument("--dev-sample-per-class", type=int, default=0, help="Optional development-only balanced subset size per class. Use 0 for full data.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-epochs", type=int, default=5)
    parser.add_argument("--improved-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--improved-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--use-mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment", default="cct_swimming_pool_cv")
    parser.add_argument("--max-error-examples", type=int, default=12)

    args = parser.parse_args()

    dev_sample_per_class = None if args.dev_sample_per_class == 0 else args.dev_sample_per_class

    if args.val_size + args.test_size >= 1.0:
        raise ValueError("val_size + test_size must be less than 1.0")

    return RunConfig(
        project_root=args.project_root,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        dev_sample_per_class=dev_sample_per_class,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        baseline_epochs=args.baseline_epochs,
        improved_epochs=args.improved_epochs,
        learning_rate=args.learning_rate,
        improved_learning_rate=args.improved_learning_rate,
        weight_decay=args.weight_decay,
        val_size=args.val_size,
        test_size=args.test_size,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        use_mlflow=args.use_mlflow,
        mlflow_experiment=args.mlflow_experiment,
        max_error_examples=args.max_error_examples,
    )


if __name__ == "__main__":
    try:
        config = parse_args()
        run_baseline_and_improved(config)
    except Exception as exc:
        print(f"Execution failed: {exc}", file=sys.stderr)
        raise
