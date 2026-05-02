"""Recursive Monte Carlo future forecasting module for wildfire probability."""

from .config import ForecastConfig
from .feature_stats import (
    compute_featurewise_homoscedastic_noise_stats,
    compute_history_window_means,
)
from .forecast_runner import run_recursive_future_forecast
from .mlp_model import FutureProbabilityMLP
from .monte_carlo_projector import (
    deterministic_recursive_future_features,
    recursive_sample_future_features_featurewise_homoscedastic,
)
from .training_data import (
    CellTargetKey,
    build_cellwise_mlp_inputs,
    build_cellwise_target,
    build_training_record,
    build_training_records,
    log_training_record_sample,
)
from .training_pipeline import (
    CellTrainingPoint,
    CellwiseMCDataset,
    StreamingMCTrainConfig,
    TrainMLPConfig,
    build_dataloader_from_records,
    build_training_records_from_cell_timeslice,
    predict_probabilities_from_records,
    select_training_points,
    train_future_probability_mlp,
    train_future_probability_mlp_streaming_mc,
)

__all__ = [
    "CellTargetKey",
    "ForecastConfig",
    "FutureProbabilityMLP",
    "build_cellwise_mlp_inputs",
    "build_cellwise_target",
    "build_training_record",
    "build_training_records",
    "log_training_record_sample",
    "CellwiseMCDataset",
    "CellTrainingPoint",
    "TrainMLPConfig",
    "StreamingMCTrainConfig",
    "build_dataloader_from_records",
    "build_training_records_from_cell_timeslice",
    "select_training_points",
    "train_future_probability_mlp",
    "train_future_probability_mlp_streaming_mc",
    "predict_probabilities_from_records",
    "compute_featurewise_homoscedastic_noise_stats",
    "compute_history_window_means",
    "deterministic_recursive_future_features",
    "recursive_sample_future_features_featurewise_homoscedastic",
    "run_recursive_future_forecast",
]
