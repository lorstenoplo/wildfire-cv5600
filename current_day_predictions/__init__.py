"""PatchTST + DLA wildfire modeling package."""

from .configs import ModelConfig, TrainConfig, DataConfig
from .losses import build_loss, FocalBCEWithLogitsLoss
from .metrics import compute_binary_metrics, find_best_threshold_by_f1
from .model_patchtst_dla import (
    PatchTSTDLAClassifier,
    TemporalMLPClassifier,
    TemporalConvClassifier,
)
from .data_io import (
    SplitArrays,
    FeatureNormalizer,
    load_split_xy_csv,
    download_hf_split_files,
    make_torch_datasets,
)
from .trainer import fit_model, evaluate_loader, evaluate_with_threshold_search, make_dataloaders

__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "PatchTSTDLAClassifier",
    "TemporalMLPClassifier",
    "TemporalConvClassifier",
    "SplitArrays",
    "FeatureNormalizer",
    "load_split_xy_csv",
    "download_hf_split_files",
    "make_torch_datasets",
    "fit_model",
    "evaluate_loader",
    "evaluate_with_threshold_search",
    "make_dataloaders",
    "build_loss",
    "FocalBCEWithLogitsLoss",
    "compute_binary_metrics",
    "find_best_threshold_by_f1",
]
