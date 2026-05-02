from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds. deterministic=True favors reproducibility over speed."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def device_summary() -> dict:
    cuda = torch.cuda.is_available()
    n_gpu = torch.cuda.device_count() if cuda else 0
    names = [torch.cuda.get_device_name(i) for i in range(n_gpu)] if cuda else []
    return {
        "cuda_available": bool(cuda),
        "n_gpu": int(n_gpu),
        "gpu_names": names,
    }
