# -*- coding: utf-8 -*-
import glob
import os
from collections import defaultdict
from typing import Dict, List, Sequence

import pandas as pd


def parse_feature_string(value) -> List[str]:
    if pd.isna(value):
        return []

    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_recorded_randk(csv_dir: str, supported_models: Sequence[str]):
    randk_records = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    csv_files = glob.glob(os.path.join(csv_dir, "*_all_results.csv"))
    supported_model_set = set(supported_models)

    for path in csv_files:
        model_name = os.path.basename(path).replace("_all_results.csv", "").lower()

        if model_name not in supported_model_set:
            continue

        df = pd.read_csv(path)
        cols_lower = {column.lower(): column for column in df.columns}

        required_cols = ["exp", "k", "repeat", "features"]
        if not all(column in cols_lower for column in required_cols):
            continue

        exp_col = cols_lower["exp"]
        k_col = cols_lower["k"]
        repeat_col = cols_lower["repeat"]
        features_col = cols_lower["features"]

        subset = df[df[exp_col].astype(str).str.lower() == "randk"].copy()

        for _, row in subset.iterrows():
            repeat = int(row[repeat_col])
            k = int(row[k_col])
            features = parse_feature_string(row[features_col])

            if len(features) == 0:
                continue

            if features not in randk_records[model_name][repeat][k]:
                randk_records[model_name][repeat][k].append(features)

    return randk_records
