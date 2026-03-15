from __future__ import annotations

import argparse
import json
import yaml
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from cv_utils.reporting import save_training_curves, save_confusion_matrix, save_threshold_plot, save_calibration_plot, save_error_panel, metrics_to_frame, dataset_summary_frame, save_html_report
from cv_utils.common import set_seed, Timer
from cv_utils.data import load_labels_dataframe, make_balanced_subset, stratified_train_val_test_split, make_dataloaders
from cv_utils.modeling import build_resnet18_binary, evaluate, select_decision_threshold, fit_model
from paths import resolve_cv_paths

# ============================================================
# CONFIG
# ============================================================


@dataclass
class RunConfig:
    """Configuration for end-to-end model training and reporting."""

    project_root: str = "."
    image_dir: str = "data/raw/swimming_pools"
    output_dir: str = "reports/cv"
    dev_sample_per_class: Optional[int] = None
    image_size: int = 224
    use_improved_augmentation: bool = True
    improved_resize_padding: int = 16
    improved_rotation_degrees: float = 10.0
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
    baseline_dropout: float = 0.2
    improved_dropout: float = 0.2
    baseline_freeze_backbone: bool = True
    improved_freeze_backbone: bool = False
    threshold_min: float = 0.20
    threshold_max: float = 0.80
    threshold_step: float = 0.05
    max_error_examples: int = 12
    report_title: str = "Swimming Pool Classification Report"
    early_stopping_patience: int = 2
    min_delta: float = 1e-4



# ============================================================
# PIPELINE
# ============================================================


def build_threshold_grid(cfg: RunConfig) -> Tuple[float, ...]:
    """Build the decision-threshold grid from configuration values."""
    return tuple(np.round(np.arange(cfg.threshold_min, cfg.threshold_max + cfg.threshold_step / 2, cfg.threshold_step), 2))


def prepare_modelling_data(cfg: RunConfig, timer: Timer) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load labels and optionally create a balanced development subset."""
    timer.start("load_labels")
    full_label_df = load_labels_dataframe(cfg)
    timer.stop("load_labels")

    if cfg.dev_sample_per_class is not None:
        timer.start("dev_sample")
        modelling_df = make_balanced_subset(full_label_df, cfg.dev_sample_per_class, cfg.seed)
        timer.stop("dev_sample")
        data_scope = f"development_subset_{cfg.dev_sample_per_class}_per_class"
    else:
        modelling_df = full_label_df.copy()
        data_scope = "full_data"

    return full_label_df, modelling_df, data_scope


def create_split_summary(
    full_label_df: pd.DataFrame,
    modelling_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_scope: str,
) -> pd.DataFrame:
    """Create a summary table describing full data, modelling source, and splits."""
    return pd.DataFrame(
        [
            dataset_summary_frame(full_label_df, "full_available"),
            dataset_summary_frame(modelling_df, f"modelling_source_{data_scope}"),
            dataset_summary_frame(train_df, "train"),
            dataset_summary_frame(val_df, "validation"),
            dataset_summary_frame(test_df, "holdout_test"),
        ]
    )


def run_baseline_stage(
    cfg: RunConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
    timer: Timer,
) -> Dict:
    """Run baseline dataloading, training, evaluation, and artifact generation."""
    evaluation_criterion = nn.BCEWithLogitsLoss()

    timer.start("baseline_dataloaders")
    baseline_train_loader, baseline_val_loader, baseline_test_loader = make_dataloaders(
        train_df,
        val_df,
        test_df,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        improved=False,
        use_improved_augmentation=cfg.use_improved_augmentation,
        improved_resize_padding=cfg.improved_resize_padding,
        improved_rotation_degrees=cfg.improved_rotation_degrees,
    )
    timer.stop("baseline_dataloaders")

    timer.start("baseline_training")
    baseline_model = build_resnet18_binary(
        pretrained=True,
        freeze_backbone=cfg.baseline_freeze_backbone,
        dropout=cfg.baseline_dropout,
    )
    baseline_model, baseline_history, baseline_meta = fit_model(
        model=baseline_model,
        train_loader=baseline_train_loader,
        val_loader=baseline_val_loader,
        device=device,
        epochs=cfg.baseline_epochs,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        early_stopping_patience=cfg.early_stopping_patience,
        min_delta=cfg.min_delta,
        stage_label="Baseline",
    )
    timer.stop("baseline_training")

    timer.start("baseline_evaluation")
    baseline_test_metrics = evaluate(baseline_model, baseline_test_loader, evaluation_criterion, device, threshold=0.5)
    timer.stop("baseline_evaluation")

    baseline_metrics_df = metrics_to_frame(baseline_test_metrics, stage="initial_solution", threshold=0.5)

    return {
        "model": baseline_model,
        "history": baseline_history,
        "meta": baseline_meta,
        "test_metrics": baseline_test_metrics,
        "metrics_df": baseline_metrics_df,
    }


def run_improved_stage(
    cfg: RunConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
    timer: Timer,
) -> Dict:
    """Run improved dataloading, training, threshold selection, evaluation, and artifact generation."""
    evaluation_criterion = nn.BCEWithLogitsLoss()

    timer.start("improved_dataloaders")
    improved_train_loader, improved_val_loader, improved_test_loader = make_dataloaders(
        train_df,
        val_df,
        test_df,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        improved=True,
        use_improved_augmentation=cfg.use_improved_augmentation,
        improved_resize_padding=cfg.improved_resize_padding,
        improved_rotation_degrees=cfg.improved_rotation_degrees,
    )
    timer.stop("improved_dataloaders")

    timer.start("improved_training")
    improved_model = build_resnet18_binary(
        pretrained=True,
        freeze_backbone=cfg.improved_freeze_backbone,
        dropout=cfg.improved_dropout,
    )
    improved_model, improved_history, improved_meta = fit_model(
        model=improved_model,
        train_loader=improved_train_loader,
        val_loader=improved_val_loader,
        device=device,
        epochs=cfg.improved_epochs,
        lr=cfg.improved_learning_rate,
        weight_decay=cfg.weight_decay,
        early_stopping_patience=cfg.early_stopping_patience,
        min_delta=cfg.min_delta,
        stage_label="Improved",
    )
    timer.stop("improved_training")

    timer.start("threshold_tuning")
    improved_val_metrics = evaluate(improved_model, improved_val_loader, evaluation_criterion, device, threshold=0.5)
    best_threshold, threshold_df = select_decision_threshold(
        improved_val_metrics["y_true"],
        improved_val_metrics["y_prob"],
        build_threshold_grid(cfg),
        selection_metric="f1",
    )

    timer.stop("threshold_tuning")

    timer.start("improved_evaluation")
    improved_test_metrics = evaluate(improved_model, improved_test_loader, evaluation_criterion, device, threshold=best_threshold)
    timer.stop("improved_evaluation")

    improved_metrics_df = metrics_to_frame(improved_test_metrics, stage="improved_solution", threshold=best_threshold)

    return {
        "model": improved_model,
        "history": improved_history,
        "meta": improved_meta,
        "val_metrics": improved_val_metrics,
        "test_metrics": improved_test_metrics,
        "metrics_df": improved_metrics_df,
        "best_threshold": best_threshold,
        "threshold_df": threshold_df,
    }


def save_pipeline_outputs(
    cfg: RunConfig,
    paths: Dict,
    baseline_results: Dict,
    improved_results: Dict,
    class_summary: pd.DataFrame,
    timings_df: pd.DataFrame,
    run_notes: list[str],
    data_scope: str,
) -> Dict:
    """Persist models, tables, figures, reports, and summary artifacts."""
    torch.save(baseline_results["model"].state_dict(), paths["models_dir"] / "baseline_resnet18.pt")
    baseline_results["history"].to_csv(paths["output_dir"] / "baseline_history.csv", index=False)
    save_training_curves(
        baseline_results["history"],
        paths["figures_dir"] / "baseline_training_curves.png",
        "Initial solution: baseline training curves",
    )
    save_confusion_matrix(
        baseline_results["test_metrics"]["y_true"],
        baseline_results["test_metrics"]["y_prob"],
        threshold=0.5,
        output_path=paths["figures_dir"] / "baseline_confusion_matrix.png",
        title="Baseline confusion matrix (hold-out test, threshold=0.50)",
    )
    baseline_results["metrics_df"].to_csv(paths["output_dir"] / "baseline_metrics.csv", index=False)

    torch.save(improved_results["model"].state_dict(), paths["models_dir"] / "improved_resnet18.pt")
    improved_results["history"].to_csv(paths["output_dir"] / "improved_history.csv", index=False)
    improved_results["threshold_df"].to_csv(paths["output_dir"] / "threshold_tuning.csv", index=False)
    save_training_curves(
        improved_results["history"],
        paths["figures_dir"] / "improved_training_curves.png",
        "Improved solution: training curves",
    )
    save_threshold_plot(improved_results["threshold_df"], paths["figures_dir"] / "threshold_tuning.png")
    save_confusion_matrix(
        improved_results["test_metrics"]["y_true"],
        improved_results["test_metrics"]["y_prob"],
        threshold=improved_results["best_threshold"],
        output_path=paths["figures_dir"] / "improved_confusion_matrix.png",
        title=f"Improved confusion matrix (hold-out test, threshold={improved_results['best_threshold']:.2f})",
    )
    save_calibration_plot(
        improved_results["test_metrics"]["y_true"],
        improved_results["test_metrics"]["y_prob"],
        paths["figures_dir"] / "improved_calibration.png",
    )
    save_error_panel(
        improved_results["test_metrics"]["y_true"],
        improved_results["test_metrics"]["y_prob"],
        improved_results["test_metrics"]["paths"],
        threshold=improved_results["best_threshold"],
        output_path=paths["figures_dir"] / "error_panel.png",
        max_examples=cfg.max_error_examples,
    )
    improved_results["metrics_df"].to_csv(paths["output_dir"] / "improved_metrics.csv", index=False)

    timings_df.to_csv(paths["output_dir"] / "timings.csv", index=False)
    class_summary.to_csv(paths["output_dir"] / "class_summary.csv", index=False)

    run_metadata = {
        "baseline_best_epoch": baseline_results["meta"].get("best_epoch"),
        "baseline_best_val_f1": baseline_results["meta"].get("best_val_f1"),
        "improved_best_epoch": improved_results["meta"].get("best_epoch"),
        "improved_best_val_f1": improved_results["meta"].get("best_val_f1"),
        "best_threshold_improved": improved_results["best_threshold"],
        "data_scope": data_scope,
    }
    (paths["output_dir"] / "model_selection_summary.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    config_path = paths["output_dir"] / "run_config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    report_path = save_html_report(
        cfg=cfg,
        class_summary=class_summary,
        timings_df=timings_df,
        baseline_metrics_df=baseline_results["metrics_df"],
        improved_metrics_df=improved_results["metrics_df"],
        threshold_df=improved_results["threshold_df"],
        paths=paths,
        best_threshold=improved_results["best_threshold"],
        run_notes=run_notes,
    )

    summary = {
        "device": run_notes[0].replace("Device used: ", ""),
        "baseline_metrics": baseline_results["metrics_df"].to_dict(orient="records"),
        "improved_metrics": improved_results["metrics_df"].to_dict(orient="records"),
        "best_threshold": improved_results["best_threshold"],
        "report_path": str(report_path),
        "timings": timings_df.set_index("step")["seconds"].to_dict() if not timings_df.empty else {},
        "output_dir": str(paths["output_dir"]),
        "config_path": str(config_path),
        "model_selection_summary": run_metadata,
    }
    with open(paths["output_dir"] / "run_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    return summary


def run_baseline_and_improved(cfg: RunConfig) -> Dict:
    """Run the full baseline and improved training pipeline."""
    set_seed(cfg.seed)
    timer = Timer()
    paths = resolve_cv_paths(project_root=cfg.project_root, image_dir=cfg.image_dir, output_dir=cfg.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_label_df, modelling_df, data_scope = prepare_modelling_data(cfg, timer)

    timer.start("split_data")
    train_df, val_df, test_df = stratified_train_val_test_split(
        modelling_df, val_size=cfg.val_size, test_size=cfg.test_size, seed=cfg.seed
    )
    timer.stop("split_data")

    class_summary = create_split_summary(full_label_df, modelling_df, train_df, val_df, test_df, data_scope)

    run_notes = [
        f"Device used: {device}",
        f"Data scope used for both baseline and improved models: {data_scope}",
        f"Fixed random seed: {cfg.seed}",
        "Hold-out test set created once and used only for final comparison.",
        f"Early stopping patience: {cfg.early_stopping_patience}",
        f"Outputs written to: {paths['output_dir']}",
    ]

    baseline_results = run_baseline_stage(cfg, train_df, val_df, test_df, device, timer)
    improved_results = run_improved_stage(cfg, train_df, val_df, test_df, device, timer)

    timings_df = timer.summary_df()
    summary = save_pipeline_outputs(
        cfg=cfg,
        paths=paths,
        baseline_results=baseline_results,
        improved_results=improved_results,
        class_summary=class_summary,
        timings_df=timings_df,
        run_notes=run_notes,
        data_scope=data_scope,
    )

    print("Run complete.")
    print(json.dumps(summary, indent=2))
    print(f"HTML report written to: {summary['report_path']}")
    return summary


# ============================================================
# CLI
# ============================================================


def load_yaml_config(config_path: str) -> dict:
    """Load run configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must define a mapping at top level: {config_path}")
    return data


def merge_config_with_cli(config_dict: dict, args: argparse.Namespace) -> dict:
    """Merge YAML config with explicit CLI overrides."""
    merged = dict(config_dict)
    cli_values = vars(args)
    for key, value in cli_values.items():
        if key == "config":
            continue
        if value is not None:
            merged[key] = value
    return merged


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Train baseline and improved swimming pool classifier.")

    parser.add_argument("--config", type=str, default="configs/cv/default.yaml")

    parser.add_argument("--project_root", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument(
        "--dev_sample_per_class",
        type=int,
        default=None,
        help="Optional development-only balanced subset size per class.",
    )
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--use_improved_augmentation", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--improved_resize_padding", type=int, default=None)
    parser.add_argument("--improved_rotation_degrees", type=float, default=None)

    parser.add_argument("--baseline_epochs", type=int, default=None)
    parser.add_argument("--improved_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--improved_learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)

    parser.add_argument("--val_size", type=float, default=None)
    parser.add_argument("--test_size", type=float, default=None)

    parser.add_argument("--baseline_dropout", type=float, default=None)
    parser.add_argument("--improved_dropout", type=float, default=None)
    parser.add_argument("--baseline_freeze_backbone", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--improved_freeze_backbone", type=str, choices=["true", "false"], default=None)

    parser.add_argument("--threshold_min", type=float, default=None)
    parser.add_argument("--threshold_max", type=float, default=None)
    parser.add_argument("--threshold_step", type=float, default=None)

    parser.add_argument("--max_error_examples", type=int, default=None)
    parser.add_argument("--report_title", type=str, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--min_delta", type=float, default=None)

    args = parser.parse_args()

    yaml_config = load_yaml_config(args.config)
    merged = merge_config_with_cli(yaml_config, args)

    if merged.get("baseline_freeze_backbone") in {"true", "false"}:
        merged["baseline_freeze_backbone"] = merged["baseline_freeze_backbone"] == "true"
    if merged.get("improved_freeze_backbone") in {"true", "false"}:
        merged["improved_freeze_backbone"] = merged["improved_freeze_backbone"] == "true"
    if merged.get("use_improved_augmentation") in {"true", "false"}:
        merged["use_improved_augmentation"] = merged["use_improved_augmentation"] == "true"


    if merged.get("dev_sample_per_class") == 0:
        merged["dev_sample_per_class"] = None

    if merged.get("val_size", 0) + merged.get("test_size", 0) >= 1.0:
        raise ValueError("val_size + test_size must be less than 1.0")
    if merged.get("threshold_step", 0) <= 0:
        raise ValueError("threshold_step must be greater than 0")
    if merged.get("threshold_min", 0) >= merged.get("threshold_max", 1):
        raise ValueError("threshold_min must be less than threshold_max")

    return RunConfig(**merged)


if __name__ == "__main__":
    try:
        config = parse_args()
        run_baseline_and_improved(config)
    except Exception as exc:
        print(f"Execution failed: {exc}", file=sys.stderr)
        raise
