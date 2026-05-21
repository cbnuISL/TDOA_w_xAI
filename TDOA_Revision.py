# -*- coding: utf-8 -*-
import os

# =========================================================
# Pin to GPU 2 only
# Must be located before importing tensorflow / xgboost / lightgbm / catboost
# Physical GPU 2 is mapped as logical GPU 0 inside the code
# =========================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import glob
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import shap

from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

warnings.filterwarnings("ignore")

# =========================================================
# TensorFlow GPU Configuration
# =========================================================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] TensorFlow visible GPUs: {gpus}")
        print("[GPU] Physical GPU 2 is mapped as logical GPU 0.")
    except RuntimeError as e:
        print(e)
else:
    print("[GPU] No GPU detected by TensorFlow.")

# =========================================================
# 0. Configuration
# =========================================================
DATA_DIR = r"/data/Public/TDOA/dataset/previous dataset"
FILE_RE = "day{}.csv"

PREV_RESULT_CSV_DIR = r"/data/Public/TDOA/KM/result/IEEE_WCL"

RESULT_DIR = r"/data/Public/TDOA/KM/result/infer_time_and_parameter"
SHAP_DIR = os.path.join(RESULT_DIR, "shap")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)

BASELINE_DAYS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
SELECT_DAYS = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]

K_LIST = [21, 19, 16, 13, 10, 7, 4, 1]

N_REPEATS = 1
TEST_SIZE = 0.2
GLOBAL_SEED = 42

TREE_SHAP_SAMPLE = 256
DNN_BG_SAMPLE = 64
DNN_EVAL_SAMPLE = 128
INFER_RUNS = 30

DNN_EPOCHS = 50
DNN_BATCH_SIZE = 512
DNN_PATIENCE = 5

ML_MODELS = ["rf", "et", "xgb", "dt", "lgbm", "cat"]
DEEP_MODELS = ["dnn"]
ALL_MODELS = ML_MODELS + DEEP_MODELS

# =========================================================
# 1. Utilities
# =========================================================
def set_global_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_ml_models(seed=42):
    return {
        # sklearn models: GPU not supported
        "rf": RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1
        ),

        # sklearn models: GPU not supported
        "et": ExtraTreesClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1
        ),

        # XGBoost: Use GPU
        # Recognized as cuda:0 due to CUDA_VISIBLE_DEVICES
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
            device="cuda"
        ),

        # sklearn models: GPU not supported
        "dt": DecisionTreeClassifier(
            criterion="entropy",
            max_features="sqrt",
            random_state=seed
        ),

        # LightGBM: Use GPU
        # Requires LightGBM to be installed with GPU support
        "lgbm": LGBMClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            device_type="gpu",
            gpu_device_id=0
        ),

        # CatBoost: Use GPU
        # Internal device mapped as "0" due to CUDA_VISIBLE_DEVICES
        "cat": CatBoostClassifier(
            loss_function="MultiClass",
            depth=6,
            learning_rate=0.05,
            iterations=500,
            verbose=False,
            random_state=seed,
            task_type="GPU",
            devices="0"
        )
    }


def make_dnn(input_dim, num_classes):
    model = Sequential([
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

        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def sample_df(df, n, seed):
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def normalize_shap_values(shap_values, n_features):
    if isinstance(shap_values, list):
        arr = np.stack([np.asarray(v) for v in shap_values], axis=0)
    elif hasattr(shap_values, "values"):
        arr = np.asarray(shap_values.values)
    else:
        arr = np.asarray(shap_values)

    if arr.ndim == 1:
        return np.abs(arr)

    feature_axes = [ax for ax, size in enumerate(arr.shape) if size == n_features]

    if not feature_axes:
        raise ValueError(
            f"Cannot identify feature axis from shape={arr.shape}, "
            f"n_features={n_features}"
        )

    feature_axis = feature_axes[-1]
    reduce_axes = tuple(ax for ax in range(arr.ndim) if ax != feature_axis)

    return np.abs(arr).mean(axis=reduce_axes)


def parse_feature_string(s):
    if pd.isna(s):
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


# =========================================================
# 2. Data Loading / Encoding
# =========================================================
def build_global_label_encoder(days):
    all_labels = []

    for day in days:
        df = pd.read_csv(os.path.join(DATA_DIR, FILE_RE.format(day))).dropna()
        all_labels.extend(df["label"].astype(str).tolist())

    le = LabelEncoder()
    le.fit(all_labels)

    return le


def load_split_data(days, seed, label_encoder):
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []

    for day in days:
        df = pd.read_csv(os.path.join(DATA_DIR, FILE_RE.format(day))).dropna()

        X = df.drop(columns=["idx", "label"], errors="ignore")
        y = label_encoder.transform(df["label"].astype(str))

        X_scaled = Normalizer().fit_transform(X)
        X_df = pd.DataFrame(X_scaled, columns=X.columns)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_df,
            y,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=y
        )

        X_train_list.append(X_tr.reset_index(drop=True))
        X_test_list.append(X_te.reset_index(drop=True))
        y_train_list.append(pd.Series(y_tr).reset_index(drop=True))
        y_test_list.append(pd.Series(y_te).reset_index(drop=True))

    return X_train_list, X_test_list, y_train_list, y_test_list


def concat_selected_features(
    X_train_list,
    X_test_list,
    y_train_list,
    y_test_list,
    selected_features
):
    X_train = pd.concat(
        [x[selected_features] for x in X_train_list],
        axis=0
    ).reset_index(drop=True)

    X_test = pd.concat(
        [x[selected_features] for x in X_test_list],
        axis=0
    ).reset_index(drop=True)

    y_train = pd.concat(y_train_list, axis=0).reset_index(drop=True)
    y_test = pd.concat(y_test_list, axis=0).reset_index(drop=True)

    return X_train, X_test, y_train, y_test


# =========================================================
# 3. Load existing randk records
# =========================================================
def load_recorded_randk(csv_dir):
    randk_records = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    csv_files = glob.glob(os.path.join(csv_dir, "*_all_results.csv"))

    for path in csv_files:
        model_name = os.path.basename(path).replace("_all_results.csv", "").lower()

        if model_name not in ALL_MODELS:
            continue

        df = pd.read_csv(path)
        cols_lower = {c.lower(): c for c in df.columns}

        required_cols = ["exp", "k", "repeat", "features"]
        if not all(c in cols_lower for c in required_cols):
            continue

        exp_col = cols_lower["exp"]
        k_col = cols_lower["k"]
        repeat_col = cols_lower["repeat"]
        features_col = cols_lower["features"]

        sub = df[df[exp_col].astype(str).str.lower() == "randk"].copy()

        for _, row in sub.iterrows():
            rep = int(row[repeat_col])
            k = int(row[k_col])
            feats = parse_feature_string(row[features_col])

            if len(feats) == 0:
                continue

            if feats not in randk_records[model_name][rep][k]:
                randk_records[model_name][rep][k].append(feats)

    return randk_records


# =========================================================
# 4. Parameter Count Calculation
# =========================================================
def count_params_ml(model, model_name):
    if model_name == "dt":
        return int(model.tree_.node_count)

    if model_name in ["rf", "et"]:
        return int(sum(est.tree_.node_count for est in model.estimators_))

    if model_name == "xgb":
        booster = model.get_booster()
        df = booster.trees_to_dataframe()
        return int(len(df))

    if model_name == "lgbm":
        dump = model.booster_.dump_model()

        def _count_nodes(node):
            if "left_child" in node and "right_child" in node:
                return 1 + _count_nodes(node["left_child"]) + _count_nodes(node["right_child"])
            return 1

        total = 0
        for tree_info in dump["tree_info"]:
            total += _count_nodes(tree_info["tree_structure"])

        return int(total)

    if model_name == "cat":
        leaf_counts = model.get_tree_leaf_counts()
        return int(sum(2 * lc - 1 for lc in leaf_counts))

    raise ValueError(f"Unsupported model_name={model_name}")


def count_params_dnn(model):
    return int(model.count_params())


# =========================================================
# 5. Measure Inference Time
# =========================================================
def measure_inference_time(model, X_test, model_name, n_runs=INFER_RUNS):
    X_np = (
        X_test.to_numpy(dtype=np.float32)
        if isinstance(X_test, pd.DataFrame)
        else np.asarray(X_test, dtype=np.float32)
    )

    if model_name == "dnn":
        with tf.device("/GPU:0"):
            _ = model.predict(X_np, verbose=0, batch_size=1024)
    else:
        _ = model.predict(X_np)

    t0 = time.perf_counter()

    for _ in range(n_runs):
        if model_name == "dnn":
            with tf.device("/GPU:0"):
                _ = model.predict(X_np, verbose=0, batch_size=1024)
        else:
            _ = model.predict(X_np)

    t1 = time.perf_counter()

    avg_total_ms = (t1 - t0) * 1000.0 / n_runs
    avg_per_sample_ms = avg_total_ms / len(X_np)

    return float(avg_total_ms), float(avg_per_sample_ms)


# =========================================================
# 6. SHAP Calculation
# =========================================================
def compute_tree_