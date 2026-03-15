from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix
)
from PIL import Image

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cct_ds_challenge.train_cv import RunConfig

# ============================================================
# REPORTING
# ============================================================


def save_training_curves(history_df: pd.DataFrame, output_path: Path, title: str) -> None:
    """Save loss and F1 curves for one training run."""
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
    """Save a confusion matrix plot at the specified threshold."""
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
    """Save a plot of validation metrics over candidate thresholds."""
    
    # Ensure thresholds are ordered before plotting
    df = threshold_df.sort_values("threshold")

    fig, ax = plt.subplots(figsize=(7, 4))

    metrics = ["accuracy", "precision", "recall", "f1"]

    for metric_name in metrics:
        ax.plot(
            df["threshold"],
            df[metric_name],
            marker="o",
            linestyle="-",
            label=metric_name,
        )

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Validation metrics across thresholds")
    ax.set_ylim(0.88, 1.0)

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)




def save_calibration_plot(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    """Save a calibration curve for predicted probabilities."""
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
    image_paths: Sequence[str],
    threshold: float,
    output_path: Path,
    max_examples: int = 12,
) -> None:
    """Save a panel of false positives and false negatives."""
    y_pred = (y_prob >= threshold).astype(int)
    false_pos_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    false_neg_idx = np.where((y_true == 1) & (y_pred == 0))[0]

    selected_indices = list(false_pos_idx[: max_examples // 2]) + list(false_neg_idx[: max_examples // 2])
    if len(selected_indices) == 0:
        print("No FP/FN examples to show.")
        return

    n_examples = len(selected_indices)
    n_cols = 4
    n_rows = math.ceil(n_examples / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, idx in zip(axes, selected_indices):
        image = Image.open(image_paths[idx]).convert("RGB")
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"true={int(y_true[idx])}, pred={int(y_pred[idx])}\np={y_prob[idx]:.2f}", fontsize=9)

    fig.suptitle("Error analysis: false positives and false negatives")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def metrics_to_frame(metrics: Dict[str, float], stage: str, threshold: float) -> pd.DataFrame:
    """Convert a metrics dictionary to a one-row DataFrame."""
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "loss"]
    rows = [{"stage": stage, "threshold": threshold, **{key: metrics.get(key) for key in metric_keys}}]
    return pd.DataFrame(rows)



def dataset_summary_frame(label_df: pd.DataFrame, name: str) -> Dict[str, object]:
    """Build a summary row for a labelled dataframe split."""
    counts = label_df["label"].value_counts().sort_index().to_dict()
    return {
        "split": name,
        "rows": len(label_df),
        "no_pool": counts.get(0, 0),
        "pool": counts.get(1, 0),
        "pool_rate": counts.get(1, 0) / len(label_df) if len(label_df) else np.nan,
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
    """Write the self-contained HTML report for the run."""
    report_path = paths["output_dir"] / "cv_report.html"

    def rel(path: Path) -> str:
        return os.path.relpath(path, report_path.parent)

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
            ul {{ margin-top: 8px; }}
        </style>
    </head>
    <body>
        <h1>{cfg.report_title}</h1>
        <p>
            This report presents a fair two-stage modelling workflow for swimming pool classification from imagery.
            Both the initial and improved solutions are trained and selected on the same train and validation splits,
            and both are compared on the same untouched hold-out test set. This makes the final comparison easier to interpret.
        </p>

        <div class='note'>
            <strong>Crux of the problem.</strong>
            Swimming pool detection from aerial imagery is not purely a modelling problem.
            Three factors interact:
            <ul>
                <li><strong>Visual ambiguity.</strong> Small pools, shadows, roofs, and water-like surfaces can produce false positives or false negatives.</li>
                <li><strong>Decision thresholds.</strong> Even a well-performing classifier requires a decision threshold to convert probabilities into labels.</li>
                <li><strong>Operational objectives.</strong> The acceptable balance between false positives and false negatives depends on how the predictions will be used operationally.</li>
            </ul>
            Importantly, the dataset is not perfectly balanced (more pools than non-pools).
            Rather than artificially correcting this imbalance during training, the model is trained on the natural data distribution.
            This preserves the meaning of the predicted probabilities and avoids distortions that can arise from resampling or loss reweighting.
            Instead, the decision threshold is selected on the validation set after training.
            This follows the general statistical principle that class imbalance in probability-based prediction problems is often better handled through threshold choice than through changing the training distribution.
            Reference: van Smeden et al., <em>"The harm of class imbalance corrections for risk prediction models."</em> 
        </div>

        <h2>1. Execution summary</h2>
        <ul>{notes_html}</ul>

        <h2>2. Data summary</h2>
        <p>
            The dataset contains more pools than non-pools. The modelling pipeline intentionally preserves this natural class distribution during training
            rather than artificially balancing the data. This is important because probability-based prediction models are most useful when their predicted
            scores remain interpretable as probabilities. The operational decision rule can then be adjusted afterwards through threshold selection.
        </p>
        {class_summary.to_html(index=False)}

        <h2>3. Initial solution</h2>
        <p><strong>Model:</strong> ResNet18 transfer learning<br>
        <strong>Preprocessing:</strong> resize to {cfg.image_size}×{cfg.image_size}, ImageNet normalization<br>
        <strong>Selection:</strong> best validation F1 with early stopping<br>
        <strong>Test evaluation:</strong> untouched hold-out test set</p>
        <p>
            The baseline uses a standard 0.50 threshold. This provides a clean reference point before introducing any threshold tuning.
            It answers the first question of the project: can a straightforward transfer-learning pipeline separate pools from non-pools under a simple setup?
        </p>
        {baseline_metrics_df.to_html(index=False)}
        <img src='{rel(baseline_curve)}' alt='Baseline training curves'>
        <img src='{rel(baseline_cm)}' alt='Baseline confusion matrix'>

        <h2>4. Improved solution</h2>
        <p><strong>Changes from baseline:</strong></p>
        <ul>
            <li>Same train, validation, and hold-out test split as the baseline</li>
            <li>Moderate augmentation rather than aggressive image distortion</li>
            <li>More trainable parameters via fine-tuning</li>
            <li>Threshold selection performed after training rather than modifying the class distribution</li>
            <li>Threshold tuning on validation probabilities</li>
            <li>Error analysis on false positives and false negatives</li>
            <li>Calibration check for prediction probabilities</li>
            <li>Early stopping based on validation F1</li>
        </ul>

        <p><strong>Selected threshold:</strong> {best_threshold:.2f}</p>
        <p>
            The model outputs probabilities for the presence of a swimming pool. A decision threshold is therefore required to convert probabilities into class predictions.
            Rather than fixing this threshold at 0.5, the threshold is selected using the validation set to maximise the F1 score.
            This allows the final decision rule to reflect the empirical distribution of predicted probabilities rather than assuming that 0.5 is always optimal.
        </p>
        <p>
            Lower thresholds typically increase recall by detecting more pools, while higher thresholds typically increase precision by reducing false alarms.
            The selected threshold therefore represents the best trade-off under the chosen evaluation metric, not a direct reflection of class prevalence.
        </p>

        {improved_metrics_df.to_html(index=False)}
        <img src='{rel(improved_curve)}' alt='Improved training curves'>
        <img src='{rel(threshold_plot)}' alt='Threshold tuning'>
        <img src='{rel(improved_cm)}' alt='Improved confusion matrix'>
        <img src='{rel(calib_plot)}' alt='Calibration plot'>
        <img src='{rel(errors_plot)}' alt='Error analysis panel'>

        <h2>5. Threshold tuning evidence</h2>
        <p>
            Threshold selection is based on validation-set probabilities only. The hold-out test set remains untouched until the final evaluation step.
            This preserves a fair separation between model selection and final performance estimation.
        </p>
        {threshold_df.to_html(index=False)}

        <h2>6. Runtime and resource awareness</h2>
        {timings_df.to_html(index=False)}

        <h2>7. Interpretation</h2>
        <p>
            The initial solution establishes a strong baseline using transfer learning and minimal assumptions.
            The improved solution focuses on reliability rather than simply increasing model complexity.
            Moderate augmentation, partial fine-tuning, and explicit threshold selection all contribute to better overall performance while keeping the comparison fair.
        </p>
        <p>
            A particularly important design choice is the separation between <strong>probability estimation</strong> and <strong>decision making</strong>.
            The model first estimates probabilities for the presence of a pool. Only afterwards is a threshold selected to convert probabilities into binary predictions.
            This makes the pipeline easier to adapt to different operational objectives.
        </p>
        <p>
            Operationally, the City may care about the balance between false negatives and false positives in different ways.
            If the objective is to identify as many potential pools as possible for follow-up inspection, a higher-recall threshold may be preferred.
            If inspection capacity is limited and false alarms are costly, a higher-precision threshold may be more appropriate.
            For this report the threshold was chosen to maximise the F1 score, which balances precision and recall, but this may not be the final operational metric.
        </p>

        <h2>8. Reproducibility</h2>
        <p>
            All outputs are written under <code>{cfg.output_dir}</code>, including figures, saved model weights, metrics, timing logs,
            the saved configuration, and this HTML report. The script is intended to run end-to-end with a single command and no manual interaction.
        </p>
    </body>
    </html>
    """
    report_path.write_text(html, encoding="utf-8")
    return report_path
