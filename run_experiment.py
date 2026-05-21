# -*- coding: utf-8 -*-
import argparse
import warnings
from typing import List

from config import ExperimentConfig
from data_utils import build_global_label_encoder
from experiment_runner import run_experiments
from model_registry import configure_tensorflow_gpu
from randk_utils import load_recorded_randk
from result_utils import (
    print_output_paths,
    save_metric_outputs,
    save_shap_summary_outputs,
)

warnings.filterwarnings("ignore")


def parse_target_models(requested_models: List[str], cfg: ExperimentConfig) -> List[str]:
    normalized = [model.lower() for model in requested_models]

    if "all" in normalized:
        return cfg.all_models

    unsupported = sorted(set(normalized) - set(cfg.all_models))
    if unsupported:
        raise ValueError(
            f"Unsupported model(s): {unsupported}. "
            f"Supported models are: {cfg.all_models}"
        )

    deduplicated = []
    for model in normalized:
        if model not in deduplicated:
            deduplicated.append(model)

    return deduplicated


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run model-wise SHAP, parameter-count, and inference-time experiments."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to run. Choose from: all, rf, et, xgb, dt, lgbm, cat, dnn.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset directory. Defaults to the path defined in ExperimentConfig.",
    )
    parser.add_argument(
        "--prev-result-csv-dir",
        default=None,
        help="Directory containing previous *_all_results.csv files for Rand-k reproduction.",
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        help="Output directory. Defaults to the path defined in ExperimentConfig.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Number of repeated experiments.",
    )
    parser.add_argument(
        "--infer-runs",
        type=int,
        default=None,
        help="Number of repeated inference runs for timing.",
    )

    return parser.parse_args()


def build_config_from_args(args) -> ExperimentConfig:
    cfg = ExperimentConfig()

    if args.data_dir is not None:
        cfg.data_dir = args.data_dir

    if args.prev_result_csv_dir is not None:
        cfg.prev_result_csv_dir = args.prev_result_csv_dir

    if args.result_dir is not None:
        cfg.result_dir = args.result_dir

    if args.n_repeats is not None:
        cfg.n_repeats = args.n_repeats

    if args.infer_runs is not None:
        cfg.infer_runs = args.infer_runs

    cfg.ensure_output_dirs()

    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)
    target_models = parse_target_models(args.models, cfg)

    print(f"[Models] {target_models}")

    configure_tensorflow_gpu()

    label_encoder = build_global_label_encoder(
        sorted(set(cfg.baseline_days + cfg.select_days)),
        cfg,
    )

    randk_records = load_recorded_randk(
        cfg.prev_result_csv_dir,
        cfg.all_models,
    )

    metrics_df, shap_all_df = run_experiments(
        target_models,
        label_encoder,
        randk_records,
        cfg,
    )

    save_metric_outputs(metrics_df, cfg)
    save_shap_summary_outputs(shap_all_df, cfg)
    print_output_paths(cfg, shap_all_df)


if __name__ == "__main__":
    main()
