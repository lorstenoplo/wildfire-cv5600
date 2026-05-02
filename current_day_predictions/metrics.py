from __future__ import annotations

import numpy as np


def _safe_import_sklearn():
    try:
        from sklearn.metrics import (
            average_precision_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )
    except Exception as e:  # pragma: no cover
        raise ImportError("scikit-learn is required for metrics") from e
    return average_precision_score, precision_score, recall_score, f1_score, confusion_matrix


def _sanitize_inputs(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_prob).reshape(-1)
    if yt.size != yp.size:
        raise ValueError(f"y_true and y_prob length mismatch: {yt.size} vs {yp.size}")

    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]

    if yt.size == 0:
        return yt.astype(np.uint8), yp.astype(np.float64)

    yt = (yt > 0.5).astype(np.uint8)
    yp = np.clip(yp.astype(np.float64), 0.0, 1.0)
    return yt, yp


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    average_precision_score, precision_score, recall_score, f1_score, confusion_matrix = _safe_import_sklearn()

    y_true, y_prob = _sanitize_inputs(y_true, y_prob)
    thr = float(threshold)

    if y_true.size == 0:
        return {
            "threshold": thr,
            "pr_auc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "iou": float("nan"),
            "specificity": float("nan"),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    y_pred = (y_prob >= thr).astype(np.uint8)

    pos = int(y_true.sum())
    neg = int(y_true.size - pos)
    if pos == 0:
        ap = 0.0
    elif neg == 0:
        ap = 1.0
    else:
        ap = float(average_precision_score(y_true, y_prob))

    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    iou = float(tp / max(tp + fp + fn, 1))
    spec = float(tn / max(tn + fp, 1))

    return {
        "threshold": thr,
        "pr_auc": ap,
        "precision": p,
        "recall": r,
        "f1": f1,
        "iou": iou,
        "specificity": spec,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def find_best_threshold_by_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    t_min: float = 0.05,
    t_max: float = 0.95,
    t_steps: int = 19,
) -> tuple[float, dict[str, float | int], list[dict[str, float | int]]]:
    y_true, y_prob = _sanitize_inputs(y_true, y_prob)
    if y_true.size == 0:
        empty = compute_binary_metrics(y_true, y_prob, threshold=0.5)
        return 0.5, empty, [empty]

    n_steps = max(2, int(t_steps))
    grid = np.linspace(float(t_min), float(t_max), n_steps)
    rows = [compute_binary_metrics(y_true, y_prob, threshold=float(t)) for t in grid]
    best = max(
        rows,
        key=lambda d: (
            float(np.nan_to_num(d["f1"], nan=-1.0)),
            float(np.nan_to_num(d["recall"], nan=-1.0)),
            float(np.nan_to_num(d["precision"], nan=-1.0)),
        ),
    )
    return float(best["threshold"]), best, rows
