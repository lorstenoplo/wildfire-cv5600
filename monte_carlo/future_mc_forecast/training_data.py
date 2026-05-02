from __future__ import annotations

import math
from datetime import date
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class CellTargetKey:
    """Unique cell target key: date + row + column."""

    date: str
    row: int
    col: int

    def as_string(self) -> str:
        return f"{self.date}_{self.row}_{self.col}"


def build_cellwise_mlp_inputs(
    projected_t_plus_h_features: torch.Tensor | np.ndarray,
    current_probability: torch.Tensor | np.ndarray | float,
    future_day_of_year: int,
) -> torch.Tensor:
    """Build MLP input X for one cell (or a batch of MC samples for one cell).

    X = [simulated_t+7_features, current_probability, sin(day), cos(day)]

    Args:
        projected_t_plus_h_features:
            shape [F] for one sample, or [N, F] for N MC samples.
        current_probability:
            scalar for the same cell at current day.
        future_day_of_year:
            integer in [1, 366].

    Returns:
        Tensor shape [N, F+3].
    """
    if not (1 <= int(future_day_of_year) <= 366):
        raise ValueError(f"future_day_of_year must be in [1, 366], got {future_day_of_year}")

    x = torch.as_tensor(projected_t_plus_h_features, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)  # [1,F]
    elif x.ndim != 2:
        raise ValueError(
            "projected_t_plus_h_features must be [F] or [N,F], "
            f"got shape={tuple(x.shape)}"
        )
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    n, _ = x.shape
    p = torch.as_tensor(
        current_probability,
        dtype=x.dtype,
        device=x.device,
    ).view(1)
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = p.expand(n, 1)

    angle = 2.0 * math.pi * float(future_day_of_year) / 365.0
    sin_day = torch.full(
        (n, 1),
        float(math.sin(angle)),
        dtype=x.dtype,
        device=x.device,
    )
    cos_day = torch.full(
        (n, 1),
        float(math.cos(angle)),
        dtype=x.dtype,
        device=x.device,
    )

    out = torch.cat([x, p, sin_day, cos_day], dim=1)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def build_cellwise_target(
    observed_t_plus_h_label_or_probability: torch.Tensor | np.ndarray | float,
) -> torch.Tensor:
    """Build supervised target Y for one cell at t+7.

    Y = real observed label/probability at t+7 for the same cell.
    """
    y = torch.as_tensor(observed_t_plus_h_label_or_probability, dtype=torch.float32).view(1)
    y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return y


def build_training_record(
    date: str,
    row: int,
    col: int,
    projected_t_plus_h_features: torch.Tensor | np.ndarray,
    current_probability: torch.Tensor | np.ndarray | float,
    future_day_of_year: int,
    observed_t_plus_h_label_or_probability: torch.Tensor | np.ndarray | float,
) -> dict[str, object]:
    """Build one cell-wise supervised training record: {key, X, Y}.

    key = date + row + column
    X   = simulated t+7 features + current probability + future temporal encoding
    Y   = real observed label/probability at t+7 for the same cell
    """
    key_obj = CellTargetKey(date=str(date), row=int(row), col=int(col))
    x = build_cellwise_mlp_inputs(
        projected_t_plus_h_features=projected_t_plus_h_features,
        current_probability=current_probability,
        future_day_of_year=future_day_of_year,
    )
    y = build_cellwise_target(
        observed_t_plus_h_label_or_probability=observed_t_plus_h_label_or_probability
    )
    return {
        "key": key_obj.as_string(),
        "X": x,
        "Y": y,
    }


def build_training_records(
    anchor_date: str,
    target_date: str,
    row: int,
    col: int,
    projected_t_plus_h_features: torch.Tensor | np.ndarray,
    current_probability_at_t: torch.Tensor | np.ndarray | float,
    future_day_of_year_t_plus_h: int | None,
    observed_t_plus_h_label_or_probability: torch.Tensor | np.ndarray | float,
    mc_id_offset: int = 0,
) -> list[dict[str, Any]]:
    """Build supervised rows for one cell across MC samples.

    Each output row has:
      key, anchor_date, target_date, row, col, mc_id, x, y
    """
    key_obj = CellTargetKey(date=str(anchor_date), row=int(row), col=int(col))
    future_doy = _resolve_target_day_of_year(
        target_date=target_date,
        future_day_of_year_t_plus_h=future_day_of_year_t_plus_h,
    )

    x = build_cellwise_mlp_inputs(
        projected_t_plus_h_features=projected_t_plus_h_features,
        current_probability=current_probability_at_t,
        future_day_of_year=future_doy,
    )
    y = float(
        build_cellwise_target(
            observed_t_plus_h_label_or_probability=observed_t_plus_h_label_or_probability
        )
        .item()
    )

    records: list[dict[str, Any]] = []
    for mc_id in range(x.shape[0]):
        records.append(
            {
                "key": key_obj.as_string(),
                "anchor_date": str(anchor_date),
                "target_date": str(target_date),
                "row": int(row),
                "col": int(col),
                "mc_id": int(mc_id_offset + mc_id),
                "x": x[mc_id].clone(),
                "y": y,
                "current_probability_source_date": str(anchor_date),
                "future_day_of_year_source_date": str(target_date),
                "future_day_of_year_t_plus_h": int(future_doy),
            }
        )
    return records


def log_training_record_sample(
    record: dict[str, Any],
    first_k_x_values: int = 6,
) -> None:
    """Print one training record summary for verification/debugging."""
    x_t = torch.as_tensor(record["x"], dtype=torch.float32).flatten()
    y_v = float(record["y"])
    print("[Training Record Sample]")
    print(f"key: {record['key']}")
    print(f"anchor_date: {record['anchor_date']}")
    print(f"target_date: {record['target_date']}")
    print(f"row: {record['row']}")
    print(f"col: {record['col']}")
    print(f"mc_id: {record['mc_id']}")
    print(f"x length: {int(x_t.numel())}")
    print(f"first few x values: {x_t[:first_k_x_values].tolist()}")
    print(f"y value: {y_v}")
    print(f"current_probability source date: {record.get('current_probability_source_date')}")
    print(f"future_day_of_year source date: {record.get('future_day_of_year_source_date')}")


def _resolve_target_day_of_year(
    target_date: str,
    future_day_of_year_t_plus_h: int | None,
) -> int:
    target = date.fromisoformat(str(target_date))
    target_doy = int(target.timetuple().tm_yday)
    if future_day_of_year_t_plus_h is None:
        return target_doy
    given = int(future_day_of_year_t_plus_h)
    if given != target_doy:
        raise ValueError(
            "future_day_of_year_t_plus_h must match target_date day-of-year. "
            f"target_date={target_date} has day_of_year={target_doy}, got={given}"
        )
    return given
