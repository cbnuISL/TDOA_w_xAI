# -*- coding: utf-8 -*-
import os
from typing import Sequence

import numpy as np
import pandas as pd
import shap
import tensorflow as tf

from config import ExperimentConfig
from model_registry import get_tf_device


def sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()

    return df.sample(n=n, random_state=seed).copy()


def normalize_shap_values(shap_values, n_features: int) -> np.ndarray:
    if isinstance(shap_values, list):
        arr = np.stack([np.asarray(value) for value in shap_values], axis=0)
    elif hasattr(shap_values, "values"):
        arr = np.asarray(shap_values.values)
    else:
        arr = np.asarray(shap_values)

    if arr.ndim == 1:
        return np.abs(arr)

    feature_axes = [axis for axis, size in enumerate(arr.shape) if size == n_features]

    if not feature_axes:
        raise ValueError(
            f"Cannot identify feature axis from shape={arr.shape}, "
            f"n_features={n_features}"
        )

    feature_axis = feature_axes[-1]
    reduce_axes = tuple(axis for axis in range(arr.ndim) if axis != feature_axis)

    return np.abs(arr).mean(axis=reduce_axes)


def compute_tree_shap_summary(
    model,
    x_train: pd.DataFrame,
    selected_features: Sequence[str],
    seed: int,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    x_eval = sample_df(
        x_train[list(selected_features)],
        cfg.tree_shap_sample,
        seed,
    )

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_eval)
    except Exception:
        explainer = shap.Explainer(model.predict, x_eval)
        shap_values = explainer(x_eval)

    mean_abs = normalize_shap_values(shap_values, len(selected_features))

    return (
        pd.DataFrame(
            {
                "feature": list(selected_features),
                "mean_abs_shap": mean_abs,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def compute_dnn_shap_summary(
    model,
    x_train: pd.DataFrame,
    selected_features: Sequence[str],
    seed: int,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    x_bg = sample_df(
        x_train[list(selected_features)],
        cfg.dnn_bg_sample,
        seed,
    ).to_numpy(dtype=np.float32)

    x_eval = sample_df(
        x_train[list(selected_features)],
        cfg.dnn_eval_sample,
        seed,
    ).to_numpy(dtype=np.float32)

    try:
        with tf.device(get_tf_device()):
            explainer = shap.GradientExplainer(model, x_bg)
            shap_values = explainer.shap_values(x_eval)
    except Exception:
        explainer = shap.Explainer(model.predict, x_bg)
        shap_values = explainer(x_eval)

    mean_abs = normalize_shap_values(shap_values, len(selected_features))

    return (
        pd.DataFrame(
            {
                "feature": list(selected_features),
                "mean_abs_shap": mean_abs,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def save_shap_outputs(
    shap_df: pd.DataFrame,
    repeat: int,
    scenario: str,
    model_name: str,
    exp: str,
    k: int,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    output = shap_df.copy()

    output.insert(0, "repeat", repeat)
    output.insert(1, "scenario", scenario)
    output.insert(2, "model", model_name)
    output.insert(3, "exp", exp)
    output.insert(4, "k", k)

    output["rank"] = np.arange(1, len(output) + 1)

    filename = f"{model_name}_{scenario}_{exp}_k{k}_repeat{repeat}_shap.csv"
    output.to_csv(
        os.path.join(cfg.shap_dir, filename),
        index=False,
        encoding="utf-8-sig",
    )

    return output
