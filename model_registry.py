# -*- coding: utf-8 -*-
import time
from typing import Dict

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from config import ExperimentConfig


def configure_tensorflow_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            print(f"[GPU] TensorFlow visible GPUs: {gpus}")
            print("[GPU] Physical GPU 2 is mapped as logical GPU 0.")
        except RuntimeError as exc:
            print(exc)
    else:
        print("[GPU] No GPU detected by TensorFlow.")


def get_tf_device() -> str:
    return "/GPU:0" if tf.config.list_logical_devices("GPU") else "/CPU:0"


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_ml_models(seed: int = 42) -> Dict[str, object]:
    return {
        # Scikit-learn model. GPU acceleration is not supported.
        "rf": RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
        ),

        # Scikit-learn model. GPU acceleration is not supported.
        "et": ExtraTreesClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
        ),

        # XGBoost GPU mode.
        # Because CUDA_VISIBLE_DEVICES is set to "2", the selected GPU is seen as cuda:0.
        "xgb": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            device="cuda",
        ),

        # Scikit-learn model. GPU acceleration is not supported.
        "dt": DecisionTreeClassifier(
            criterion="entropy",
            max_features="sqrt",
            random_state=seed,
        ),

        # LightGBM GPU mode.
        # LightGBM must be installed with GPU support.
        "lgbm": LGBMClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            device_type="gpu",
            gpu_device_id=0,
        ),

        # CatBoost GPU mode.
        # Because CUDA_VISIBLE_DEVICES is set to "2", the selected GPU is seen as device "0".
        "cat": CatBoostClassifier(
            loss_function="MultiClass",
            depth=6,
            learning_rate=0.05,
            iterations=500,
            verbose=False,
            random_state=seed,
            task_type="GPU",
            devices="0",
        ),
    }


def make_dnn(input_dim: int, num_classes: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(2048, activation="relu", kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dropout(0.1),
            Dense(1024, activation="relu", kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dense(512, activation="relu", kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dense(256, activation="relu", kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_ml_model(model_name: str, x_train: pd.DataFrame, y_train: pd.Series, seed: int):
    models = make_ml_models(seed)

    if model_name not in models:
        raise ValueError(f"Unsupported ML model: {model_name}")

    model = models[model_name]
    model.fit(x_train, y_train.values.ravel())

    return model


def train_dnn_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    num_classes: int,
    seed: int,
    cfg: ExperimentConfig,
):
    set_global_seed(seed)

    with tf.device(get_tf_device()):
        model = make_dnn(x_train.shape[1], num_classes)

        early_stopping = EarlyStopping(
            monitor="loss",
            patience=cfg.dnn_patience,
            restore_best_weights=True,
        )

        model.fit(
            x_train.to_numpy(dtype=np.float32),
            y_train.to_numpy(),
            epochs=cfg.dnn_epochs,
            batch_size=cfg.dnn_batch_size,
            verbose=0,
            callbacks=[early_stopping],
        )

    return model


def count_params_ml(model, model_name: str) -> int:
    if model_name == "dt":
        return int(model.tree_.node_count)

    if model_name in ["rf", "et"]:
        return int(sum(estimator.tree_.node_count for estimator in model.estimators_))

    if model_name == "xgb":
        booster = model.get_booster()
        tree_df = booster.trees_to_dataframe()
        return int(len(tree_df))

    if model_name == "lgbm":
        dump = model.booster_.dump_model()

        def count_nodes(node) -> int:
            if "left_child" in node and "right_child" in node:
                return 1 + count_nodes(node["left_child"]) + count_nodes(node["right_child"])
            return 1

        total = 0
        for tree_info in dump["tree_info"]:
            total += count_nodes(tree_info["tree_structure"])

        return int(total)

    if model_name == "cat":
        leaf_counts = model.get_tree_leaf_counts()
        return int(sum(2 * leaf_count - 1 for leaf_count in leaf_counts))

    raise ValueError(f"Unsupported model_name={model_name}")


def count_params_dnn(model) -> int:
    return int(model.count_params())


def measure_inference_time(
    model,
    x_test,
    model_name: str,
    cfg: ExperimentConfig,
):
    x_np = (
        x_test.to_numpy(dtype=np.float32)
        if isinstance(x_test, pd.DataFrame)
        else np.asarray(x_test, dtype=np.float32)
    )

    if model_name == "dnn":
        with tf.device(get_tf_device()):
            _ = model.predict(x_np, verbose=0, batch_size=1024)
    else:
        _ = model.predict(x_np)

    start = time.perf_counter()

    for _ in range(cfg.infer_runs):
        if model_name == "dnn":
            with tf.device(get_tf_device()):
                _ = model.predict(x_np, verbose=0, batch_size=1024)
        else:
            _ = model.predict(x_np)

    end = time.perf_counter()

    avg_total_ms = (end - start) * 1000.0 / cfg.infer_runs
    avg_per_sample_ms = avg_total_ms / len(x_np)

    return float(avg_total_ms), float(avg_per_sample_ms)
