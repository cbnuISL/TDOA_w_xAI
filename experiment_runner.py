# -*- coding: utf-8 -*-
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import ExperimentConfig
from data_utils import concat_selected_features, load_split_data
from model_registry import (
    count_params_dnn,
    count_params_ml,
    measure_inference_time,
    train_dnn_model,
    train_ml_model,
)
from shap_utils import (
    compute_dnn_shap_summary,
    compute_tree_shap_summary,
    save_shap_outputs,
)


def build_shap_rankings_for_selected_days(
    seed: int,
    label_encoder,
    target_models: Sequence[str],
    cfg: ExperimentConfig,
) -> Dict[str, List[str]]:
    x_train_list, x_test_list, y_train_list, y_test_list = load_split_data(
        cfg.select_days,
        seed,
        label_encoder,
        cfg,
    )

    all_features = x_train_list[0].columns.tolist()

    x_train_all, x_test_all, y_train_all, y_test_all = concat_selected_features(
        x_train_list,
        x_test_list,
        y_train_list,
        y_test_list,
        all_features,
    )

    rankings = {}
    ranking_csv_rows = []

    for model_name in target_models:
        print(f"[SHAP Ranking] model={model_name}")

        if model_name in cfg.ml_models:
            model = train_ml_model(
                model_name,
                x_train_all,
                y_train_all,
                seed,
            )

            shap_df = compute_tree_shap_summary(
                model,
                x_train_all,
                all_features,
                seed,
                cfg,
            )
        elif model_name in cfg.deep_models:
            model = train_dnn_model(
                x_train_all,
                y_train_all,
                len(label_encoder.classes_),
                seed,
                cfg,
            )

            shap_df = compute_dnn_shap_summary(
                model,
                x_train_all,
                all_features,
                seed,
                cfg,
            )
        else:
            raise ValueError(f"Unsupported model_name={model_name}")

        rankings[model_name] = shap_df["feature"].tolist()

        for rank_idx, row in shap_df.reset_index(drop=True).iterrows():
            ranking_csv_rows.append(
                {
                    "repeat": seed - cfg.global_seed + 1,
                    "model": model_name,
                    "feature": row["feature"],
                    "mean_abs_shap": row["mean_abs_shap"],
                    "rank": rank_idx + 1,
                }
            )

    ranking_df = pd.DataFrame(ranking_csv_rows)

    ranking_df.to_csv(
        os.path.join(
            cfg.result_dir,
            f"repeat_{seed - cfg.global_seed + 1}_selected_days_shap_ranking.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    return rankings


def run_single_experiment(
    repeat: int,
    scenario: str,
    model_name: str,
    exp: str,
    k: int,
    selected_features: Sequence[str],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int,
    num_classes: int,
    cfg: ExperimentConfig,
):
    print(
        f"[Run] repeat={repeat}, scenario={scenario}, "
        f"model={model_name}, exp={exp}, k={k}"
    )

    if model_name in cfg.ml_models:
        model = train_ml_model(
            model_name,
            x_train,
            y_train,
            seed,
        )

        param_count = count_params_ml(model, model_name)

        infer_total_ms, infer_per_sample_ms = measure_inference_time(
            model,
            x_test,
            model_name,
            cfg,
        )

        shap_df = compute_tree_shap_summary(
            model,
            x_train,
            selected_features,
            seed,
            cfg,
        )
    elif model_name in cfg.deep_models:
        model = train_dnn_model(
            x_train,
            y_train,
            num_classes,
            seed,
            cfg,
        )

        param_count = count_params_dnn(model)

        infer_total_ms, infer_per_sample_ms = measure_inference_time(
            model,
            x_test,
            "dnn",
            cfg,
        )

        shap_df = compute_dnn_shap_summary(
            model,
            x_train,
            selected_features,
            seed,
            cfg,
        )
    else:
        raise ValueError(f"Unsupported model_name={model_name}")

    shap_output = save_shap_outputs(
        shap_df,
        repeat,
        scenario,
        model_name,
        exp,
        k,
        cfg,
    )

    metric_row = {
        "repeat": repeat,
        "scenario": scenario,
        "model": model_name,
        "exp": exp,
        "k": int(k),
        "n_features": int(len(selected_features)),
        "param_count": int(param_count),
        "inference_time_total_ms": float(infer_total_ms),
        "inference_time_per_sample_ms": float(infer_per_sample_ms),
        "n_train_samples": int(len(x_train)),
        "n_test_samples": int(len(x_test)),
        "selected_features": ",".join(selected_features),
    }

    return metric_row, shap_output


def run_experiments(
    target_models: Sequence[str],
    label_encoder,
    randk_records,
    cfg: ExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_metric_rows = []
    all_shap_rows = []

    for repeat in range(1, cfg.n_repeats + 1):
        seed = cfg.global_seed + (repeat - 1)

        print(f"\n[Repeat {repeat}/{cfg.n_repeats}] seed={seed}")

        x_train_baseline_list, x_test_baseline_list, y_train_baseline_list, y_test_baseline_list = (
            load_split_data(
                cfg.baseline_days,
                seed,
                label_encoder,
                cfg,
            )
        )

        x_train_selected_list, x_test_selected_list, y_train_selected_list, y_test_selected_list = (
            load_split_data(
                cfg.select_days,
                seed,
                label_encoder,
                cfg,
            )
        )

        baseline_all_features = x_train_baseline_list[0].columns.tolist()
        selected_all_features = x_train_selected_list[0].columns.tolist()

        shap_rankings = build_shap_rankings_for_selected_days(
            seed,
            label_encoder,
            target_models,
            cfg,
        )

        x_train_baseline, x_test_baseline, y_train_baseline, y_test_baseline = (
            concat_selected_features(
                x_train_baseline_list,
                x_test_baseline_list,
                y_train_baseline_list,
                y_test_baseline_list,
                baseline_all_features,
            )
        )

        # -----------------------------
        # Baseline experiments
        # -----------------------------
        for model_name in target_models:
            row, shap_rows = run_single_experiment(
                repeat=repeat,
                scenario="baseline_day4_included",
                model_name=model_name,
                exp="baseline",
                k=len(baseline_all_features),
                selected_features=baseline_all_features,
                x_train=x_train_baseline,
                x_test=x_test_baseline,
                y_train=y_train_baseline,
                y_test=y_test_baseline,
                seed=seed,
                num_classes=len(label_encoder.classes_),
                cfg=cfg,
            )

            all_metric_rows.append(row)
            all_shap_rows.append(shap_rows)

        # -----------------------------
        # Top-k and bottom-k experiments
        # -----------------------------
        for model_name in target_models:
            ranking = shap_rankings[model_name]

            for k in cfg.k_list:
                if k > len(ranking):
                    continue

                top_features = ranking[:k]

                x_train_top, x_test_top, y_train_top, y_test_top = concat_selected_features(
                    x_train_selected_list,
                    x_test_selected_list,
                    y_train_selected_list,
                    y_test_selected_list,
                    top_features,
                )

                row, shap_rows = run_single_experiment(
                    repeat=repeat,
                    scenario="feature_selected_day4_excluded",
                    model_name=model_name,
                    exp="topk",
                    k=k,
                    selected_features=top_features,
                    x_train=x_train_top,
                    x_test=x_test_top,
                    y_train=y_train_top,
                    y_test=y_test_top,
                    seed=seed,
                    num_classes=len(label_encoder.classes_),
                    cfg=cfg,
                )

                all_metric_rows.append(row)
                all_shap_rows.append(shap_rows)

                bottom_features = ranking[-k:]

                x_train_bottom, x_test_bottom, y_train_bottom, y_test_bottom = (
                    concat_selected_features(
                        x_train_selected_list,
                        x_test_selected_list,
                        y_train_selected_list,
                        y_test_selected_list,
                        bottom_features,
                    )
                )

                row, shap_rows = run_single_experiment(
                    repeat=repeat,
                    scenario="feature_selected_day4_excluded",
                    model_name=model_name,
                    exp="bottomk",
                    k=k,
                    selected_features=bottom_features,
                    x_train=x_train_bottom,
                    x_test=x_test_bottom,
                    y_train=y_train_bottom,
                    y_test=y_test_bottom,
                    seed=seed,
                    num_classes=len(label_encoder.classes_),
                    cfg=cfg,
                )

                all_metric_rows.append(row)
                all_shap_rows.append(shap_rows)

        # -----------------------------
        # Rand-k experiments reproduced from previous CSV files
        # -----------------------------
        for model_name in target_models:
            model_randk = randk_records.get(model_name, {})
            repeat_randk = model_randk.get(repeat, {})

            for _, feature_lists in repeat_randk.items():
                for record_idx, recorded_features in enumerate(feature_lists, start=1):
                    valid_features = [
                        feature
                        for feature in recorded_features
                        if feature in selected_all_features
                    ]

                    if len(valid_features) == 0:
                        continue

                    x_train_rand, x_test_rand, y_train_rand, y_test_rand = (
                        concat_selected_features(
                            x_train_selected_list,
                            x_test_selected_list,
                            y_train_selected_list,
                            y_test_selected_list,
                            valid_features,
                        )
                    )

                    row, shap_rows = run_single_experiment(
                        repeat=repeat,
                        scenario="randk_reproduced_from_csv",
                        model_name=model_name,
                        exp="randk",
                        k=len(valid_features),
                        selected_features=valid_features,
                        x_train=x_train_rand,
                        x_test=x_test_rand,
                        y_train=y_train_rand,
                        y_test=y_test_rand,
                        seed=seed,
                        num_classes=len(label_encoder.classes_),
                        cfg=cfg,
                    )

                    row["randk_record_index"] = record_idx
                    shap_rows["randk_record_index"] = record_idx

                    all_metric_rows.append(row)
                    all_shap_rows.append(shap_rows)

    metrics_df = pd.DataFrame(all_metric_rows)

    if len(all_shap_rows) > 0:
        shap_all_df = pd.concat(
            all_shap_rows,
            axis=0,
            ignore_index=True,
        )
    else:
        shap_all_df = pd.DataFrame()

    return metrics_df, shap_all_df
