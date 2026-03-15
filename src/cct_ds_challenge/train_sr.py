from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from paths import CITY_HEX_POLYGONS_8, DATA_PROCESSED, REPORTS_DIR


RANDOM_SEED = 42
TARGET_COL = "sewer_requests"
MIN_BUILDING_COUNT = 5


def load_features() -> pd.DataFrame:
    path = DATA_PROCESSED / "sewer_hex_features.csv"
    df = pd.read_csv(path)
    df["h3_level8_index"] = df["h3_level8_index"].astype(str)
    return df


def make_model_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Restrict analysis to hexes with a minimum built environment footprint.
    # This defines a more realistic population at risk for sewer incidents and
    # avoids large numbers of structural-zero hexes such as mountains or water.
    out = out.loc[out["building_count"] >= MIN_BUILDING_COUNT].copy()

    # Avoid target leakage by removing sewer requests from total request volume.
    out["non_target_requests"] = (out["total_requests"] - out["sewer_requests"]).clip(lower=0)

    # Approximate diversity excluding the target type.
    out["diversity_ex_target"] = out["request_diversity"] - (out["sewer_requests"] > 0).astype(int)
    out["diversity_ex_target"] = out["diversity_ex_target"].clip(lower=0)

    # Log-transform skewed predictors for stability and interpretability.
    out["log_non_target_requests"] = np.log1p(out["non_target_requests"])
    out["log_building_count"] = np.log1p(out["building_count"])
    out["log_mean_building_area"] = np.log1p(out["mean_building_area"])
    out["log_sd_building_area"] = np.log1p(out["sd_building_area"])

    return out.reset_index(drop=True)


def get_feature_cols() -> list[str]:
    return [
        "log_non_target_requests",
        "diversity_ex_target",
        "log_building_count",
        "log_mean_building_area",
        "log_sd_building_area",
    ]


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=RANDOM_SEED)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, 1e-9, None)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mpd = float(mean_poisson_deviance(y_true, y_pred))
    spear = float(spearmanr(y_true, y_pred).statistic)
    return {
        "mae": mae,
        "rmse": rmse,
        "mean_poisson_deviance": mpd,
        "spearman": spear,
    }


def fit_baseline_poisson(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[Pipeline, dict[str, float]]:
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].values
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL].values

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("poisson", PoissonRegressor(alpha=0.0, max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    metrics = evaluate_predictions(y_val, val_pred)
    return model, metrics


def fit_improved_nb_glm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[sm.GLM, dict[str, float], pd.DataFrame]:
    X_train = sm.add_constant(train_df[feature_cols], has_constant="add")
    y_train = train_df[TARGET_COL].values

    nb_model = sm.GLM(
        y_train,
        X_train,
        family=sm.families.NegativeBinomial(),
    ).fit()

    X_val = sm.add_constant(val_df[feature_cols], has_constant="add")
    y_val = val_df[TARGET_COL].values
    val_pred = nb_model.predict(X_val)

    metrics = evaluate_predictions(y_val, val_pred)

    coef_df = pd.DataFrame(
        {
            "feature": nb_model.params.index,
            "coefficient": nb_model.params.values,
            "incidence_rate_ratio": np.exp(nb_model.params.values),
            "p_value": nb_model.pvalues.values,
        }
    ).sort_values("coefficient", ascending=False)

    return nb_model, metrics, coef_df


def refit_and_test(
    baseline_model: Pipeline,
    feature_cols: list[str],
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[Pipeline, dict[str, float], np.ndarray]:
    X_train_val = train_val_df[feature_cols]
    y_train_val = train_val_df[TARGET_COL].values
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].values

    baseline_model.fit(X_train_val, y_train_val)
    test_pred = baseline_model.predict(X_test)
    return baseline_model, evaluate_predictions(y_test, test_pred), test_pred


def refit_nb_and_test(
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[sm.GLM, dict[str, float], pd.DataFrame, np.ndarray]:
    X_train_val = sm.add_constant(train_val_df[feature_cols], has_constant="add")
    y_train_val = train_val_df[TARGET_COL].values

    nb_model = sm.GLM(
        y_train_val,
        X_train_val,
        family=sm.families.NegativeBinomial(),
    ).fit()

    X_test = sm.add_constant(test_df[feature_cols], has_constant="add")
    y_test = test_df[TARGET_COL].values
    test_pred = nb_model.predict(X_test)

    coef_df = pd.DataFrame(
        {
            "feature": nb_model.params.index,
            "coefficient": nb_model.params.values,
            "incidence_rate_ratio": np.exp(nb_model.params.values),
            "p_value": nb_model.pvalues.values,
        }
    ).sort_values("coefficient", ascending=False)

    metrics = evaluate_predictions(y_test, test_pred)
    return nb_model, metrics, coef_df, test_pred


def save_scatter(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.6)
    lim = max(float(np.max(y_true)), float(np.max(y_pred))) + 1
    ax.plot([0, lim], [0, lim], linestyle="--")
    ax.set_xlabel("Actual sewer requests")
    ax.set_ylabel("Predicted sewer requests")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_residual_hist(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=30)
    ax.set_title("Residual distribution")
    ax.set_xlabel("Actual - predicted")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_residual_map(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    hex_gdf = gpd.read_file(CITY_HEX_POLYGONS_8)
    hex_id_col = "index" if "index" in hex_gdf.columns else "h3_index"
    hex_gdf[hex_id_col] = hex_gdf[hex_id_col].astype(str)

    plot_df = test_df[["h3_level8_index", TARGET_COL]].copy()
    plot_df["predicted"] = y_pred
    plot_df["residual"] = plot_df[TARGET_COL] - plot_df["predicted"]

    map_df = hex_gdf.merge(
        plot_df,
        left_on=hex_id_col,
        right_on="h3_level8_index",
        how="left",
    )

    fig, ax = plt.subplots(figsize=(7, 9))
    map_df.plot(column="residual", legend=True, ax=ax)
    ax.set_title("Improved model residuals on hold-out test hexes")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_html_report(
    output_dir: Path,
    data_summary: pd.DataFrame,
    baseline_metrics_val: dict[str, float],
    baseline_metrics_test: dict[str, float],
    improved_metrics_val: dict[str, float],
    improved_metrics_test: dict[str, float],
    coef_df: pd.DataFrame,
    runtime_rows: list[dict[str, float]],
    feature_cols: list[str],
) -> None:
    runtime_df = pd.DataFrame(runtime_rows)

    top_pos = coef_df[coef_df["feature"] != "const"].sort_values("coefficient", ascending=False).head(5)
    top_neg = coef_df[coef_df["feature"] != "const"].sort_values("coefficient", ascending=True).head(5)

    html = f"""
    <html>
    <head>
        <meta charset='utf-8'>
        <title>Sewer Request Introspection Report</title>
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
        <h1>Sewer Request Introspection Report</h1>
        <p>This report presents a two-stage modelling workflow for explaining and predicting the number of sewer blockage / overflow requests per H3 level 8 hex. Both the initial and improved solutions are trained and selected on the same fixed train and validation splits, and both are compared on the same untouched hold-out test set.</p>

        <div class='note'>
            <strong>Crux of the problem.</strong> The task is not only to predict sewer request counts, but also to explain what drives them. The modelling strategy therefore uses count models that are stable, interpretable, and appropriate for overdispersed municipal request data.
        </div>

        <h2>1. Execution summary</h2>
        <ul>
            <li>Fixed random seed: {RANDOM_SEED}</li>
            <li>Target type: Sewer: Blocked/Overflow</li>
            <li>Modelling unit: H3 level 8 hex</li>
            <li>Minimum building count for inclusion: {MIN_BUILDING_COUNT}</li>
            <li>Features used: {", ".join(feature_cols)}</li>
            <li>Outputs written to: {output_dir}</li>
        </ul>

        <h2>2. Modelling scope and assumptions</h2>
        <p>The analysis is primarily explanatory rather than purely predictive. The aim is to identify variables associated with higher sewer incident counts across the city. Given the limited feature set available in this assessment, the estimated relationships should be interpreted as associations rather than causal effects, because unobserved infrastructure and environmental factors may confound the observed relationships.</p>
        <p>Hexes with fewer than {MIN_BUILDING_COUNT} buildings were excluded from modelling. This defines a more realistic population at risk for sewer incidents and reduces the influence of structural-zero areas such as mountains, water, or other non-residential zones with little or no built environment.</p>
        <p>The model explains cross-sectional variation in annual sewer request counts across hexes. It does not attempt to identify short-term causal shocks, such as the COVID-related reporting disruption observed in the exploratory analysis.</p>

        <h2>3. Data summary</h2>
        {data_summary.to_html(index=False)}

        <h2>4. Initial solution</h2>
        <p><strong>Model:</strong> Poisson regression<br>
        <strong>Reason:</strong> Simple, stable count model and a strong baseline for municipal request counts.</p>
        {pd.DataFrame([{"stage": "initial_solution_validation", **baseline_metrics_val}, {"stage": "initial_solution_test", **baseline_metrics_test}]).to_html(index=False)}
        <img src='figures/baseline_actual_vs_predicted.png' alt='Baseline actual vs predicted'>

        <h2>5. Improved solution</h2>
        <p><strong>Model:</strong> Negative Binomial GLM<br>
        <strong>Reason:</strong> Same explanatory structure as the baseline, but better suited to overdispersed count data.</p>
        {pd.DataFrame([{"stage": "improved_solution_validation", **improved_metrics_val}, {"stage": "improved_solution_test", **improved_metrics_test}]).to_html(index=False)}
        <img src='figures/improved_actual_vs_predicted.png' alt='Improved actual vs predicted'>
        <img src='figures/improved_residual_hist.png' alt='Improved residual histogram'>
        <img src='figures/improved_residual_map.png' alt='Improved residual map'>

        <h2>6. Driver analysis</h2>
        <p>For the improved model, coefficients are interpreted through incidence-rate ratios. Values above 1 imply that higher feature values are associated with higher expected sewer request counts, holding other variables fixed.</p>
        <h3>Positive drivers</h3>
        {top_pos.to_html(index=False)}
        <h3>Negative drivers</h3>
        {top_neg.to_html(index=False)}

        <h2>7. Interpretation</h2>
        <p>The initial solution establishes a transparent baseline. The improved solution keeps the modelling story interpretable while relaxing the Poisson variance assumption. This improves suitability for municipal request counts and gives a clearer basis for discussing likely drivers of sewer incidents.</p>
        <p>Because the model omits several important municipal and physical drivers, including sewer network topology, pipe age, maintenance history, rainfall, slope, and detailed population exposure, any driver statements should be interpreted cautiously. Residual spatial structure would suggest that these omitted factors still explain meaningful variation.</p>

        <h2>8. Runtime and resource awareness</h2>
        {runtime_df.to_html(index=False)}

        <h2>9. Reproducibility</h2>
        <p>All outputs are written under <code>{output_dir}</code>, including metrics, figures, predictions, coefficients, timing logs, and this HTML report. The script is intended to run end-to-end with a single command and no manual interaction.</p>
    </body>
    </html>
    """

    (output_dir / "sr_report.html").write_text(html, encoding="utf-8")


def main() -> None:
    report_dir = REPORTS_DIR / "sr"
    figures_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    runtime_rows: list[dict[str, float]] = []

    t0 = time.time()
    df = load_features()
    df = make_model_table(df)
    feature_cols = get_feature_cols()
    runtime_rows.append({"step": "load_and_prepare_features", "seconds": time.time() - t0})

    t0 = time.time()
    train_df, val_df, test_df = split_data(df)
    runtime_rows.append({"step": "split_data", "seconds": time.time() - t0})

    data_summary = pd.DataFrame(
        [
            {"split": "full_available", "rows": len(df), "mean_target": df[TARGET_COL].mean(), "zero_rate": (df[TARGET_COL] == 0).mean()},
            {"split": "train", "rows": len(train_df), "mean_target": train_df[TARGET_COL].mean(), "zero_rate": (train_df[TARGET_COL] == 0).mean()},
            {"split": "validation", "rows": len(val_df), "mean_target": val_df[TARGET_COL].mean(), "zero_rate": (val_df[TARGET_COL] == 0).mean()},
            {"split": "holdout_test", "rows": len(test_df), "mean_target": test_df[TARGET_COL].mean(), "zero_rate": (test_df[TARGET_COL] == 0).mean()},
        ]
    )

    t0 = time.time()
    baseline_model, baseline_metrics_val = fit_baseline_poisson(train_df, val_df, feature_cols)
    runtime_rows.append({"step": "baseline_training_and_validation", "seconds": time.time() - t0})

    t0 = time.time()
    improved_model_val, improved_metrics_val, coef_df_val = fit_improved_nb_glm(train_df, val_df, feature_cols)
    runtime_rows.append({"step": "improved_training_and_validation", "seconds": time.time() - t0})

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    t0 = time.time()
    baseline_model, baseline_metrics_test, baseline_test_pred = refit_and_test(
        baseline_model, feature_cols, train_val_df, test_df
    )
    runtime_rows.append({"step": "baseline_refit_and_test", "seconds": time.time() - t0})

    t0 = time.time()
    improved_model_test, improved_metrics_test, coef_df_test, improved_test_pred = refit_nb_and_test(
        train_val_df, test_df, feature_cols
    )
    runtime_rows.append({"step": "improved_refit_and_test", "seconds": time.time() - t0})

    save_scatter(
        test_df[TARGET_COL].values,
        baseline_test_pred,
        figures_dir / "baseline_actual_vs_predicted.png",
        "Baseline actual vs predicted",
    )
    save_scatter(
        test_df[TARGET_COL].values,
        improved_test_pred,
        figures_dir / "improved_actual_vs_predicted.png",
        "Improved actual vs predicted",
    )
    save_residual_hist(
        test_df[TARGET_COL].values,
        improved_test_pred,
        figures_dir / "improved_residual_hist.png",
    )
    save_residual_map(
        test_df,
        improved_test_pred,
        figures_dir / "improved_residual_map.png",
    )

    test_predictions = test_df[["h3_level8_index", TARGET_COL]].copy()
    test_predictions["baseline_prediction"] = baseline_test_pred
    test_predictions["improved_prediction"] = improved_test_pred
    test_predictions["improved_residual"] = test_predictions[TARGET_COL] - test_predictions["improved_prediction"]
    test_predictions.to_csv(report_dir / "test_predictions.csv", index=False)

    coef_df_test.to_csv(report_dir / "improved_model_coefficients.csv", index=False)
    data_summary.to_csv(report_dir / "data_summary.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(report_dir / "runtime_log.csv", index=False)

    save_html_report(
        output_dir=report_dir,
        data_summary=data_summary,
        baseline_metrics_val=baseline_metrics_val,
        baseline_metrics_test=baseline_metrics_test,
        improved_metrics_val=improved_metrics_val,
        improved_metrics_test=improved_metrics_test,
        coef_df=coef_df_test,
        runtime_rows=runtime_rows,
        feature_cols=feature_cols,
    )


if __name__ == "__main__":
    main()
