# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# =========================================================
# GPU visibility must be configured before importing
# TensorFlow, XGBoost, LightGBM, or CatBoost.
#
# Physical GPU 2 is exposed to this process only.
# Inside the process, it is mapped as logical GPU 0.
# =========================================================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@dataclass
class ExperimentConfig:
    # -----------------------------
    # Dataset and output paths
    # -----------------------------
    data_dir: str = r"/data/Public/TDOA/dataset/previous dataset"
    file_pattern: str = "day{}.csv"
    prev_result_csv_dir: str = r"/data/Public/TDOA/KM/result/IEEE_WCL"
    result_dir: str = r"/data/Public/TDOA/KM/result/infer_time_and_parameter"

    # -----------------------------
    # Day split settings
    # -----------------------------
    baseline_days: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    )
    select_days: List[int] = field(
        default_factory=lambda: [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]
    )

    # -----------------------------
    # Experiment settings
    # -----------------------------
    k_list: List[int] = field(default_factory=lambda: [21, 19, 16, 13, 10, 7, 4, 1])
    n_repeats: int = 1
    test_size: float = 0.2
    global_seed: int = 42

    # -----------------------------
    # SHAP and inference settings
    # -----------------------------
    tree_shap_sample: int = 256
    dnn_bg_sample: int = 64
    dnn_eval_sample: int = 128
    infer_runs: int = 30

    # -----------------------------
    # DNN training settings
    # -----------------------------
    dnn_epochs: int = 50
    dnn_batch_size: int = 512
    dnn_patience: int = 5

    # -----------------------------
    # Supported model groups
    # -----------------------------
    ml_models: List[str] = field(
        default_factory=lambda: ["rf", "et", "xgb", "dt", "lgbm", "cat"]
    )
    deep_models: List[str] = field(default_factory=lambda: ["dnn"])

    @property
    def all_models(self) -> List[str]:
        return self.ml_models + self.deep_models

    @property
    def shap_dir(self) -> str:
        return str(Path(self.result_dir) / "shap")

    def ensure_output_dirs(self) -> None:
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)
        Path(self.shap_dir).mkdir(parents=True, exist_ok=True)
