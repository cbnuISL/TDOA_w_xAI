# Model-wise Experiment Execution Code

## Structure

- `config.py`: Path, repeat count, GPU, SHAP, and DNN training configurations
- `data_utils.py`: Data loading, normalization, train/test split, and feature concatenation
- `model_registry.py`: RF, ET, XGB, DT, LGBM, CatBoost, and DNN creation, training, parameter counting, and inference time measurement
- `shap_utils.py`: SHAP computation and output saving
- `randk_utils.py`: Loading previously recorded Rand-k CSV results
- `experiment_runner.py`: Experiment loop for baseline, top-k, bottom-k, and rand-k scenarios
- `result_utils.py`: Saving metric results and SHAP summaries
- `run_experiment.py`: Entry point for model-wise experiment execution

## Execution Examples

Run all models:

```bash
python run_experiment.py --models all