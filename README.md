# Model-wise TDOA Feature Optimization Experiment

This repository provides a model-wise experiment pipeline for evaluating
SHAP-based feature optimization in TDOA-based localization.

The code compares multiple machine learning and deep learning models using
different feature selection strategies, including full-feature baseline,
SHAP-based top-k selection, bottom-k selection, and reproduced random-k
selection.

The main goal is to verify whether localization performance can be maintained
while reducing the number of TDOA input features.

---

## 1. Overview

In TDOA-based localization, each input feature represents a time difference
between a pair of beams or signal paths. Since multiple beams generate many
pairwise TDOA values, the input feature vector can become high-dimensional.

However, not every TDOA feature contributes equally to prediction. Some
features may be informative, while others may be redundant, unstable, or noisy.

This experiment follows the basic idea below:

```text
Load TDOA dataset
        |
        v
Train a model with all features
        |
        v
Compute SHAP values from the trained model
        |
        v
Rank features by importance
        |
        v
Select top-k / bottom-k / random-k features
        |
        v
Retrain the model with selected features
        |
        v
Compare parameter count, inference time, and SHAP results
```

---

## 2. What This Code Does

This code performs experiments for each selected model.

Supported models:

```text
rf    : Random Forest
et    : ExtraTrees
xgb   : XGBoost
dt    : Decision Tree
lgbm  : LightGBM
cat   : CatBoost
dnn   : Deep Neural Network
```

Each model can be executed independently. For example, users can run only
Decision Tree, only CatBoost, or multiple models together.

---

## 3. How SHAP Is Used

SHAP is used after the first model training step.

The process is:

```text
1. Train the model using all available TDOA features.
2. Compute SHAP values from the trained model.
3. Convert SHAP values into feature importance scores.
4. Sort features by mean absolute SHAP value.
5. Select important features based on the ranking.
6. Retrain the model using only the selected feature subset.
```

In this code, SHAP is not used as a prediction model.  
It is used only to measure how much each input feature contributes to the
model prediction.

The final feature importance score is based on:

```text
mean absolute SHAP value
```

A larger mean absolute SHAP value means that the feature has a stronger
influence on the model output.

---

## 4. Experiment Types

This code evaluates four experiment types.

### 4.1 Baseline

The baseline uses all available features.

```text
exp = baseline
```

This result is used as the reference for comparison.

---

### 4.2 Top-k

Top-k uses the most important features selected by SHAP.

```text
exp = topk
```

This experiment checks whether the model can maintain performance using only
highly important features.

---

### 4.3 Bottom-k

Bottom-k uses the least important features based on SHAP ranking.

```text
exp = bottomk
```

This experiment checks whether low-importance features are less useful for
prediction.

---

### 4.4 Rand-k

Rand-k uses previously recorded random feature subsets.

```text
exp = randk
```

This experiment is used as a comparison against SHAP-based selection.  
It verifies whether performance improvement comes from selecting meaningful
features, not simply from reducing the number of features.

---

## 5. Repository Structure

```text
.
├── config.py
├── data_utils.py
├── model_registry.py
├── shap_utils.py
├── randk_utils.py
├── experiment_runner.py
├── result_utils.py
├── run_experiment.py
└── README.md
```

### `config.py`

Defines global experiment settings.

Main settings include:

- dataset path
- result output path
- GPU configuration
- baseline days
- feature selection days
- selected k values
- repeat count
- SHAP sample size
- inference measurement runs

---

### `data_utils.py`

Handles dataset loading and preprocessing.

Main roles:

- load CSV files by day
- remove missing values
- separate features and labels
- encode labels
- normalize input features
- split train/test data
- concatenate selected features

---

### `model_registry.py`

Defines model-related functions.

Main roles:

- create each model
- train selected model
- count model parameters or tree nodes
- measure inference time

This file allows the experiment runner to handle different models through the
same interface.

---

### `shap_utils.py`

Handles SHAP computation and SHAP result saving.

Main roles:

- compute SHAP values for tree-based models
- compute SHAP values for DNN
- calculate mean absolute SHAP importance
- save per-experiment SHAP results

---

### `randk_utils.py`

Loads previously recorded random-k feature selections from CSV files.

This is used to reproduce random feature selection experiments.

---

### `experiment_runner.py`

Controls the main experiment loop.

Main roles:

- run baseline experiments
- generate SHAP feature rankings
- run top-k experiments
- run bottom-k experiments
- run rand-k experiments
- collect metric and SHAP results

---

### `result_utils.py`

Saves final metric and SHAP summary files.

Main outputs include:

```text
experiment_metrics_all.csv
experiment_metrics_summary.csv
experiment_shap_all.csv
experiment_shap_summary.csv
```

---

### `run_experiment.py`

The main entry point.

This script parses command-line arguments and runs the selected model
experiments.

---

## 6. How to Run

Run all models:

```bash
python run_experiment.py --models all
```

Run only Decision Tree:

```bash
python run_experiment.py --models dt
```

Run only CatBoost:

```bash
python run_experiment.py --models cat
```

Run CatBoost and DNN:

```bash
python run_experiment.py --models cat dnn
```

Run DT, CatBoost, and XGBoost:

```bash
python run_experiment.py --models dt cat xgb
```

Run with a custom result directory:

```bash
python run_experiment.py \
  --models dt cat \
  --result-dir /data/Public/TDOA/KM/result/modelwise_test
```

Run with a custom repeat count and inference measurement count:

```bash
python run_experiment.py \
  --models dt cat dnn \
  --n-repeats 3 \
  --infer-runs 50
```

---

## 7. Output Files

After execution, the following files are generated.

### `experiment_metrics_all.csv`

Contains all individual experiment results.

Main information:

- model name
- experiment type
- selected k value
- number of selected features
- parameter count
- inference time
- selected feature list

---

### `experiment_metrics_summary.csv`

Contains summarized metric results grouped by model, experiment type, and k.

---

### `experiment_shap_all.csv`

Contains all SHAP importance results from individual experiments.

---

### `experiment_shap_summary.csv`

Contains summarized SHAP importance results.

---

### Individual SHAP CSV Files

Each experiment also saves an individual SHAP result file under the SHAP output
directory.

Example file name:

```text
cat_feature_selected_day4_excluded_topk_k19_repeat1_shap.csv
```

---

## 8. GPU Notes

This code is configured to use physical GPU 2:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
```

Inside the process, physical GPU 2 is mapped as logical GPU 0.

Model-specific GPU usage:

```text
RF, ET, DT      : CPU
XGBoost         : GPU
LightGBM        : GPU
CatBoost        : GPU
DNN             : TensorFlow GPU
```

LightGBM GPU execution requires a GPU-enabled LightGBM installation.

---

## 9. Recommended Analysis

After running the experiments, compare the results in the following order:

```text
1. baseline vs topk
2. topk vs bottomk
3. topk vs randk
4. model-wise comparison under the same k
```

The most important comparison is:

```text
baseline vs topk vs randk
```

If `topk` maintains similar performance to the baseline while using fewer
features, it means that SHAP-based feature selection successfully preserved
important TDOA information.

If `topk` performs better than `randk`, it means that selecting important
features is more effective than arbitrary feature reduction.

---

## 10. Summary

This code is designed to test whether SHAP-based feature selection can reduce
the dimensionality of TDOA features while preserving localization performance.

The core principle is:

```text
Train with all features
        ->
Explain feature importance using SHAP
        ->
Select important features
        ->
Retrain with selected features
        ->
Compare efficiency and prediction behavior
```

This allows each model to be evaluated under the same feature optimization
strategy.
