# -*- coding: utf-8 -*-
import os
from typing import List, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Normalizer

from config import ExperimentConfig


def build_global_label_encoder(days: Sequence[int], cfg: ExperimentConfig) -> LabelEncoder:
    all_labels = []

    for day in days:
        path = os.path.join(cfg.data_dir, cfg.file_pattern.format(day))
        df = pd.read_csv(path).dropna()
        all_labels.extend(df["label"].astype(str).tolist())

    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    return label_encoder


def load_split_data(
    days: Sequence[int],
    seed: int,
    label_encoder: LabelEncoder,
    cfg: ExperimentConfig,
):
    x_train_list, x_test_list = [], []
    y_train_list, y_test_list = [], []

    for day in days:
        path = os.path.join(cfg.data_dir, cfg.file_pattern.format(day))
        df = pd.read_csv(path).dropna()

        x = df.drop(columns=["idx", "label"], errors="ignore")
        y = label_encoder.transform(df["label"].astype(str))

        x_scaled = Normalizer().fit_transform(x)
        x_df = pd.DataFrame(x_scaled, columns=x.columns)

        x_train, x_test, y_train, y_test = train_test_split(
            x_df,
            y,
            test_size=cfg.test_size,
            random_state=seed,
            stratify=y,
        )

        x_train_list.append(x_train.reset_index(drop=True))
        x_test_list.append(x_test.reset_index(drop=True))
        y_train_list.append(pd.Series(y_train).reset_index(drop=True))
        y_test_list.append(pd.Series(y_test).reset_index(drop=True))

    return x_train_list, x_test_list, y_train_list, y_test_list


def concat_selected_features(
    x_train_list: List[pd.DataFrame],
    x_test_list: List[pd.DataFrame],
    y_train_list: List[pd.Series],
    y_test_list: List[pd.Series],
    selected_features: Sequence[str],
):
    x_train = pd.concat(
        [x[selected_features] for x in x_train_list],
        axis=0,
    ).reset_index(drop=True)

    x_test = pd.concat(
        [x[selected_features] for x in x_test_list],
        axis=0,
    ).reset_index(drop=True)

    y_train = pd.concat(y_train_list, axis=0).reset_index(drop=True)
    y_test = pd.concat(y_test_list, axis=0).reset_index(drop=True)

    return x_train, x_test, y_train, y_test
