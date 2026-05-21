# -*- coding: utf-8 -*-
import os

import pandas as pd

from config import ExperimentConfig


def save_metric_outputs(metrics_df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    metrics_path = os.path.join(cfg.result_dir, "experiment_metrics_all.csv")
    summary_path = os.path.join(cfg.result_dir, "experiment_metrics_summary.csv")

    metrics_df.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = (
        metrics_df.groupby(
            ["scenario", "model", "exp", "k", "n_features"],
            as_index=False,
        )
        .agg(
            param_count_mean=("param_count", "mean"),
            param_count_std=("param_count", "std"),
            inference_time_total_ms_mean=("inference_time_total_ms", "mean"),
            inference_time_total_ms_std=("inference_time_total_ms", "std"),
            inference_time_per_sample_ms_mean=("inference_time_per_sample_ms", "mean"),
            inference_time_per_sample_ms_std=("inference_time_per_sample_ms", "std"),
            n_train_samples_mean=("n_train_samples", "mean"),
            n_test_samples_mean=("n_test_samples", "mean"),
        )
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    return summary_df


def save_shap_summary_outputs(shap_all_df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    shap_all_path = os.path.join(cfg.result_dir, "experiment_shap_all.csv")
    shap_summary_path = os.path.join(cfg.result_dir, "experiment_shap_summary.csv")

    if shap_all_df.empty:
        return pd.DataFrame()

    shap_all_df.to_csv(
        shap_all_path,
        index=False,
        encoding="utf-8-sig",
    )

    shap_summary_df = (
        shap_all_df.groupby(
            ["scenario", "model", "exp", "k", "feature"],
            as_index=False,
        )
        .agg(
            mean_abs_shap_mean=("mean_abs_shap", "mean"),
            mean_abs_shap_std=("mean_abs_shap", "std"),
            best_rank_mean=("rank", "mean"),
        )
        .sort_values(
            ["scenario", "model", "exp", "k", "mean_abs_shap_mean"],
            ascending=[True, True, True, True, False],
        )
        .reset_index(drop=True)
    )

    shap_summary_df.to_csv(
        shap_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    return shap_summary_df


def print_output_paths(cfg: ExperimentConfig, shap_all_df: pd.DataFrame) -> None:
    print("\n[Done]")
    print(os.path.join(cfg.result_dir, "experiment_metrics_all.csv"))
    print(os.path.join(cfg.result_dir, "experiment_metrics_summary.csv"))

    if not shap_all_df.empty:
        print(os.path.join(cfg.result_dir, "experiment_shap_all.csv"))
        print(os.path.join(cfg.result_dir, "experiment_shap_summary.csv"))

    print(os.path.join(cfg.shap_dir, "*.csv"))
