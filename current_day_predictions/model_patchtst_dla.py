from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import ModelConfig


class PatchTokenizer(nn.Module):
    """Tokenize ``[B,T,F]`` into channel-independent temporal patches.

    Output tokens are arranged as ``[B*F, Np, D]`` where ``Np`` is number of
    temporal patches per feature channel.
    """

    def __init__(self, seq_len: int, patch_len: int, patch_stride: int, d_model: int, dropout: float):
        super().__init__()
        self.seq_len = int(seq_len)
        self.patch_len = int(max(1, patch_len))
        self.patch_stride = int(max(1, patch_stride))
        self.d_model = int(d_model)

        self.proj = nn.Linear(self.patch_len, self.d_model)
        n_patches = max(1, 1 + (max(self.seq_len, self.patch_len) - self.patch_len) // self.patch_stride)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches, self.d_model))
        self.drop = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        # x: [B, T, F]
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B,T,F], got {tuple(x.shape)}")

        b, t, f = x.shape
        x = x.transpose(1, 2)  # [B, F, T]

        if t < self.patch_len:
            pad = self.patch_len - t
            x = F.pad(x, (pad, 0), mode="replicate")
            t = self.patch_len

        patches = x.unfold(dimension=2, size=self.patch_len, step=self.patch_stride)  # [B,F,Np,Pl]
        b, f, npat, _ = patches.shape

        tok = self.proj(patches)  # [B,F,Np,D]
        tok = tok + self.pos_emb[:, :npat, :].unsqueeze(1)
        tok = self.drop(tok)
        tok = tok.reshape(b * f, npat, self.d_model)
        return tok, b, f


class TransformerPatchEncoder(nn.Module):
    """Stacked transformer encoder; returns pooled representation per depth.

    For each layer, pooled feature output has shape ``[B,F,D]``.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.n_heads,
                    dim_feedforward=cfg.d_model * cfg.ff_mult,
                    dropout=cfg.dropout,
                    activation="gelu",
                    layer_norm_eps=cfg.layer_norm_eps,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(cfg.n_layers)
            ]
        )

    def forward(self, tokens: torch.Tensor, batch_size: int, n_features: int) -> list[torch.Tensor]:
        # tokens: [B*F, Np, D]
        h = tokens
        out: list[torch.Tensor] = []
        for layer in self.layers:
            h = layer(h)
            pooled = h.mean(dim=1).reshape(batch_size, n_features, -1)  # [B,F,D]
            out.append(pooled)
        return out


class FusionNode(nn.Module):
    """Pairwise fusion node for hierarchical DLA aggregation."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([a, b], dim=-1))


class DLAAggregator(nn.Module):
    """Deep Layer Aggregation over transformer depth outputs."""

    def __init__(self, n_layers: int, d_model: int, dropout: float):
        super().__init__()
        max_depth = max(1, math.ceil(math.log2(max(1, n_layers))) + 1)
        self.fusions = nn.ModuleList([FusionNode(d_model=d_model, dropout=dropout) for _ in range(max_depth)])

    def forward(self, layer_feats: list[torch.Tensor]) -> torch.Tensor:
        if not layer_feats:
            raise ValueError("layer_feats is empty")

        cur = layer_feats
        depth = 0
        while len(cur) > 1:
            nxt: list[torch.Tensor] = []
            i = 0
            while i < len(cur):
                if i + 1 < len(cur):
                    z = self.fusions[min(depth, len(self.fusions) - 1)](cur[i], cur[i + 1])
                    nxt.append(z)
                    i += 2
                else:
                    nxt.append(cur[i])
                    i += 1
            cur = nxt
            depth += 1
        return cur[0]  # [B,F,D]


class DLABasicBlock1D(nn.Module):
    """Residual basic block for temporal Conv1D processing."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class DLARoot1D(nn.Module):
    """Root node that merges temporal tree branches."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(xs, dim=1)
        return F.relu(self.bn(self.conv(x)), inplace=True)


class DLATree1D(nn.Module):
    """Recursive temporal DLA tree module over Conv1D blocks."""

    def __init__(
        self,
        block: type[DLABasicBlock1D],
        in_channels: int,
        out_channels: int,
        level: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        self.level = int(level)
        if self.level <= 0:
            raise ValueError("level must be >= 1")

        if self.level == 1:
            self.left = block(in_channels, out_channels, stride=stride)
            self.right = block(out_channels, out_channels, stride=1)
            self.root = DLARoot1D(in_channels=2 * out_channels, out_channels=out_channels)
        else:
            self.tree1 = DLATree1D(block, in_channels, out_channels, level=self.level - 1, stride=stride)
            self.tree2 = DLATree1D(block, out_channels, out_channels, level=self.level - 1, stride=1)
            self.root = DLARoot1D(in_channels=2 * out_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.level == 1:
            x1 = self.left(x)
            x2 = self.right(x1)
            return self.root([x1, x2])
        x1 = self.tree1(x)
        x2 = self.tree2(x1)
        return self.root([x1, x2])


class DLATreeBranch1D(nn.Module):
    """Temporal DLA branch over ``[B,F,T]`` to model lag dynamics."""

    def __init__(self, in_channels: int, out_dim: int, base_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        c0 = int(base_channels)
        c1 = int(base_channels * 2)
        c2 = int(base_channels * 4)

        self.base = nn.Sequential(
            nn.Conv1d(in_channels, c0, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(c0),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv1d(c0, c0, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(c0),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(c0, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
        )
        self.layer3 = DLATree1D(DLABasicBlock1D, c1, c1, level=1, stride=1)
        self.layer4 = DLATree1D(DLABasicBlock1D, c1, c2, level=2, stride=1)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c2, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,F,T]
        h = self.base(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        return self.head(h)


class TemporalAttentionPooling(nn.Module):
    """Learned attention pooling over window-level embeddings."""

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B,T,F]
        h = self.proj(x)  # [B,T,D]
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)  # [B,T]
        pooled = torch.sum(h * w.unsqueeze(-1), dim=1)  # [B,D]
        return pooled, w


class ResidualFeatureMixer(nn.Module):
    """Residual cross-feature mixer applied independently per time window."""

    def __init__(self, n_features: int, hidden_mult: int = 2, dropout: float = 0.05):
        super().__init__()
        h = max(int(n_features), int(n_features * max(1, hidden_mult)))
        self.norm = nn.LayerNorm(n_features)
        self.net = nn.Sequential(
            nn.Linear(n_features, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, n_features),
        )
        self.gain = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,F]
        z = self.net(self.norm(x))
        return x + torch.tanh(self.gain) * z


class TabularShortcutBranch(nn.Module):
    """Shortcut MLP over flattened + temporal-stat tabular features."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class GatedFusion(nn.Module):
    """Fuse heterogeneous branches with optional learned gates."""

    def __init__(self, input_dims: list[int], hidden_dim: int, dropout: float = 0.1, use_gates: bool = True):
        super().__init__()
        if not input_dims:
            raise ValueError("input_dims must not be empty")
        self.use_gates = bool(use_gates)
        self.n_sources = len(input_dims)

        self.proj = nn.ModuleList([nn.Linear(int(d), int(hidden_dim)) for d in input_dims])
        self.post = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))

        cat_dim = int(sum(input_dims))
        if self.use_gates:
            self.gate_net = nn.Linear(cat_dim, self.n_sources)
            self.fallback_proj = None
        else:
            self.gate_net = None
            self.fallback_proj = nn.Sequential(
                nn.Linear(cat_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

    def forward(self, sources: list[torch.Tensor]) -> torch.Tensor:
        if len(sources) != self.n_sources:
            raise ValueError(f"Expected {self.n_sources} sources, got {len(sources)}")

        cat = torch.cat(sources, dim=-1)
        if self.use_gates:
            gates = torch.sigmoid(self.gate_net(cat))  # [B,S]
            projected = [p(s) for p, s in zip(self.proj, sources)]
            stack = torch.stack(projected, dim=1)  # [B,S,H]
            fused = torch.sum(stack * gates.unsqueeze(-1), dim=1)  # [B,H]
        else:
            fused = self.fallback_proj(cat)

        return self.post(fused)


class PatchTSTDLAClassifier(nn.Module):
    """PatchTST + temporal DLA hybrid for cell-level wildfire risk.

    Input:
      - x: ``[B,T,F]`` where T is causal lag windows and F is feature count.
    Output:
      - logits: ``[B]``
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.n_features <= 0:
            raise ValueError("ModelConfig.n_features must be set > 0")
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(f"d_model={cfg.d_model} must be divisible by n_heads={cfg.n_heads}")

        self.cfg = cfg
        self.patchtst_only = bool(getattr(cfg, "patchtst_only", False))

        self.tokenizer = PatchTokenizer(
            seq_len=cfg.seq_len,
            patch_len=cfg.patch_len,
            patch_stride=cfg.patch_stride,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
        )
        self.encoder = TransformerPatchEncoder(cfg)
        self.dla = DLAAggregator(n_layers=cfg.n_layers, d_model=cfg.d_model, dropout=cfg.dropout)

        self.use_window_embedding = bool(getattr(cfg, "use_window_embedding", True))
        if self.use_window_embedding:
            self.window_embed = nn.Embedding(cfg.seq_len, cfg.n_features)
            nn.init.trunc_normal_(self.window_embed.weight, std=0.02)
        else:
            self.window_embed = None

        self.use_feature_mixer = (not self.patchtst_only) and bool(getattr(cfg, "use_feature_mixer", True))
        self.feature_mixer = (
            ResidualFeatureMixer(
                n_features=cfg.n_features,
                hidden_mult=int(getattr(cfg, "feature_mixer_hidden_mult", 2)),
                dropout=float(getattr(cfg, "feature_mixer_dropout", 0.05)),
            )
            if self.use_feature_mixer
            else None
        )

        self.use_temporal_dla1d = (not self.patchtst_only) and bool(getattr(cfg, "use_temporal_dla1d", True))
        self.temporal_dla1d = (
            DLATreeBranch1D(
                in_channels=cfg.n_features,
                out_dim=cfg.d_model,
                base_channels=int(getattr(cfg, "temporal_dla_base_channels", 32)),
                dropout=cfg.dropout,
            )
            if self.use_temporal_dla1d
            else None
        )

        self.use_temporal_attention_pool = (not self.patchtst_only) and bool(getattr(cfg, "use_temporal_attention_pool", True))
        self.temporal_attn_pool = (
            TemporalAttentionPooling(n_features=cfg.n_features, d_model=cfg.d_model, dropout=cfg.dropout)
            if self.use_temporal_attention_pool
            else None
        )

        self.use_summary_branches = (not self.patchtst_only) and bool(getattr(cfg, "use_summary_branches", True))
        self.use_tabular_shortcut = (not self.patchtst_only) and bool(getattr(cfg, "use_tabular_shortcut", True))

        self.feature_attn = nn.Linear(cfg.d_model, 1)

        self.recent_proj = nn.Sequential(nn.Linear(cfg.n_features, cfg.mla_hidden), nn.GELU(), nn.Dropout(cfg.dropout))
        self.mean_proj = nn.Sequential(nn.Linear(cfg.n_features, cfg.mla_hidden), nn.GELU(), nn.Dropout(cfg.dropout))
        self.std_proj = nn.Sequential(nn.Linear(cfg.n_features, cfg.mla_hidden), nn.GELU(), nn.Dropout(cfg.dropout))
        self.trend_proj = nn.Sequential(nn.Linear(cfg.n_features, cfg.mla_hidden), nn.GELU(), nn.Dropout(cfg.dropout))
        self.recent_delta_proj = nn.Sequential(
            nn.Linear(cfg.n_features, cfg.mla_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.accel_proj = nn.Sequential(nn.Linear(cfg.n_features, cfg.mla_hidden), nn.GELU(), nn.Dropout(cfg.dropout))

        tab_in_dim = (cfg.seq_len + 8) * cfg.n_features
        self.tabular_shortcut = (
            TabularShortcutBranch(
                in_dim=tab_in_dim,
                out_dim=cfg.d_model,
                hidden_dim=int(getattr(cfg, "tabular_hidden", 256)),
                dropout=cfg.dropout,
            )
            if self.use_tabular_shortcut
            else None
        )

        source_dims: list[int] = [cfg.d_model]  # transformer pooled
        if self.use_temporal_dla1d:
            source_dims.append(cfg.d_model)
        if self.use_temporal_attention_pool:
            source_dims.append(cfg.d_model)
        if self.use_tabular_shortcut:
            source_dims.append(cfg.d_model)
        if self.use_summary_branches:
            source_dims.extend([cfg.mla_hidden] * 6)

        self.fusion = GatedFusion(
            input_dims=source_dims,
            hidden_dim=int(getattr(cfg, "fusion_hidden", 128)),
            dropout=cfg.dropout,
            use_gates=(not self.patchtst_only) and bool(getattr(cfg, "use_gated_fusion", True)),
        )

        self.head = nn.Sequential(
            nn.Linear(int(getattr(cfg, "fusion_hidden", 128)), cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

    def _apply_window_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self.window_embed is None:
            return x
        b, t, f = x.shape
        if f != self.cfg.n_features:
            return x
        idx = torch.arange(t, device=x.device)
        idx = torch.clamp(idx, max=self.cfg.seq_len - 1)
        w = self.window_embed(idx).unsqueeze(0)  # [1,T,F]
        return x + w

    def _build_tabular_view(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,F]
        t = x.shape[1]
        flat = x.reshape(x.shape[0], -1)
        mean_x = x.mean(dim=1)
        std_x = x.std(dim=1, unbiased=False)
        min_x = x.min(dim=1).values
        max_x = x.max(dim=1).values
        last = x[:, -1, :]
        trend = x[:, -1, :] - x[:, 0, :]

        if t >= 2:
            recent_delta = x[:, -1, :] - x[:, -2, :]
        else:
            recent_delta = torch.zeros_like(last)

        if t >= 3:
            accel = x[:, -1, :] - 2.0 * x[:, -2, :] + x[:, -3, :]
        else:
            accel = torch.zeros_like(last)

        return torch.cat(
            [flat, mean_x, std_x, min_x, max_x, last, trend, recent_delta, accel],
            dim=-1,
        )

    def _summary_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        # x: [B,T,F]
        t = x.shape[1]

        recent = self.recent_proj(x[:, -1, :])
        mean_x = self.mean_proj(x.mean(dim=1))
        std_x = self.std_proj(x.std(dim=1, unbiased=False))
        long_trend = self.trend_proj(x[:, -1, :] - x[:, 0, :])

        if t >= 2:
            recent_delta = self.recent_delta_proj(x[:, -1, :] - x[:, -2, :])
        else:
            recent_delta = self.recent_delta_proj(torch.zeros_like(x[:, -1, :]))

        if t >= 3:
            accel = self.accel_proj(x[:, -1, :] - 2.0 * x[:, -2, :] + x[:, -3, :])
        else:
            accel = self.accel_proj(torch.zeros_like(x[:, -1, :]))

        return [recent, mean_x, std_x, long_trend, recent_delta, accel]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,F]
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,F], got {tuple(x.shape)}")

        x_raw = x.float()
        x_model = self.feature_mixer(x_raw) if self.feature_mixer is not None else x_raw
        x_model = self._apply_window_embedding(x_model)

        tokens, b, f = self.tokenizer(x_model)
        layer_feats = self.encoder(tokens=tokens, batch_size=b, n_features=f)
        feat_map = self.dla(layer_feats)  # [B,F,D]

        feat_attn = torch.softmax(self.feature_attn(feat_map).squeeze(-1), dim=1)  # [B,F]
        transformer_vec = torch.sum(feat_map * feat_attn.unsqueeze(-1), dim=1)  # [B,D]

        sources: list[torch.Tensor] = [transformer_vec]

        if self.temporal_dla1d is not None:
            # Temporal-only branch on [B,F,T].
            temporal_dla_vec = self.temporal_dla1d(x_model.transpose(1, 2))
            sources.append(temporal_dla_vec)

        if self.temporal_attn_pool is not None:
            temporal_attn_vec, _ = self.temporal_attn_pool(x_model)  # [B,D]
            sources.append(temporal_attn_vec)

        if self.tabular_shortcut is not None:
            tabular_vec = self.tabular_shortcut(self._build_tabular_view(x_raw))
            sources.append(tabular_vec)

        if self.use_summary_branches:
            # Intentionally compute summary stats on raw normalized features
            # before adding learnable window embeddings.
            sources.extend(self._summary_features(x_raw))

        fused = self.fusion(sources)
        logits = self.head(fused).squeeze(-1)  # [B]
        return logits


class TemporalMLPClassifier(nn.Module):
    """Baseline MLP on flattened ``[T,F]`` input."""

    def __init__(self, n_features: int, seq_len: int = 4, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        inp = int(n_features) * int(seq_len)
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,F], got {tuple(x.shape)}")
        z = x.reshape(x.shape[0], -1)
        return self.net(z).squeeze(-1)


class TemporalConvClassifier(nn.Module):
    """Baseline temporal Conv1D model: input ``[B,T,F]`` -> output ``[B]``."""

    def __init__(self, n_features: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=2, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=2, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,F], got {tuple(x.shape)}")
        z = x.transpose(1, 2)  # [B,F,T]
        h = self.conv(z).mean(dim=2)
        return self.head(h).squeeze(-1)
