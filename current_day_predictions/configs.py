from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataConfig:
    expected_windows: int = 4
    batch_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    drop_last_train: bool = False


@dataclass
class ModelConfig:
    seq_len: int = 4
    patch_len: int = 2
    patch_stride: int = 1
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    attn_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    n_features: int = 0  # set at runtime after loading data

    # Summary / head dims
    mla_hidden: int = 128
    head_hidden: int = 128

    # Backward compatibility field (legacy name). Not used by the new model path.
    use_tree_dla2d: bool = True
    dla2d_base_channels: int = 32

    # New temporal architecture toggles
    use_temporal_dla1d: bool = True
    temporal_dla_base_channels: int = 32
    use_window_embedding: bool = True
    use_temporal_attention_pool: bool = True
    use_summary_branches: bool = True
    use_gated_fusion: bool = True
    fusion_hidden: int = 128

    # Cross-feature residual mixer over each window [T,F].
    use_feature_mixer: bool = True
    feature_mixer_hidden_mult: int = 2
    feature_mixer_dropout: float = 0.05

    # Shortcut branch on flattened + temporal-stat tabular view.
    use_tabular_shortcut: bool = True
    tabular_hidden: int = 256

    # When True, disables auxiliary branches and keeps transformer path only.
    # This is useful for strict PatchTST-only experiments.
    patchtst_only: bool = False


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    patience: int = 6
    use_amp: bool = True

    # Loss
    loss_name: str = "bce"  # choices: bce, focal
    pos_weight: float = 1.0
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0  # reserved for future, kept for research config compatibility

    # Validation threshold sweep
    threshold_grid_min: float = 0.05
    threshold_grid_max: float = 0.95
    threshold_grid_steps: int = 19

    # Model selection / LR scheduling
    monitor_metric: str = "pr_auc"  # choices: val_loss, pr_auc, f1, iou
    scheduler_name: str = "cosine"  # choices: none, cosine
    min_lr: float = 1e-5
