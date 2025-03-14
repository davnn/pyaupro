from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import Tensor


def auc_compute(
        x: Tensor,
        y: Tensor,
        limit: float = 1.0,
        *,
        descending: bool = False,
        reorder: bool = False,
        check: bool = True,
        return_curve: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """Compute area under the curve using the trapezoidal rule.

    Args:
        x:
            Ascending (or descending if ``descending=True``) sorted vector if,
            otherwise ``reorder`` must be used.
        y:
            Vector of the same size as ``x``.
        limit:
            Integration limit chosen for ``x`` such that only the values until
            the limit are used for computation.
        descending:
            Input vector ``x`` is descending or ``reorder`` sorts descending.
        check:
            Check if the given vector is monotonically increasing or decreasing.
        return_curve:
            Return the final tensors used to compute the area under the curve.

    Example:
        >>> import torch
        >>> x = torch.tensor([1, 2, 3, 4])
        >>> y = torch.tensor([1, 2, 3, 4])
        >>> _auc_compute(x, y)
        tensor(7.5000)

    """
    assert limit > 0, "The `limit` parameter must be > 0."

    with torch.no_grad():
        if reorder:
            x, x_idx = torch.sort(x, stable=True, descending=descending)
            y = y[x_idx]

        if check and not reorder:
            dx = x[1:] - x[:-1]
            if descending:
                assert (dx <= 0).all(), "The `x` tensor is not descending."
            else:
                assert (dx >= 0).all(), "The `x` tensor is not ascending."

        if limit != 1.0:
            # searchsorted expects a monotonically increasing tensor
            x_sorted = x.flip() if descending else x
            limit_idx = torch.searchsorted(x_sorted, limit)
            x, y = x[:limit_idx], y[:limit_idx]

        direction = -1.0 if descending else 1.0
        auc_score = torch.trapz(y, x) * direction
        auc_score = auc_score / limit
        return (auc_score, x, y) if return_curve else auc_score


def generate_random_data(
        batch_size: int = 8,
        height: int = 32,
        width: int = 32,
        num_objects: int = 2,
        noise_level: float =0.25,
        return_numpy: bool = False,
        seed: Any = None, # dynamically validated in ``default_rng
    ) -> tuple[Tensor, Tensor] | tuple[np.ndarray, np.ndarray]:
    """Generates random test data: binary masks and probabilistic predictions."""
    rng = np.random.default_rng(seed)
    preds = np.zeros((batch_size, height, width), dtype=np.float32)
    masks = np.zeros((batch_size, height, width), dtype=np.int32)

    for i in range(batch_size):
        # Generate random object positions
        object_centers = rng.integers(0, (height, width), size=(num_objects, 2))

        # Create binary mask with objects
        for y, x in object_centers:
            masks[i, max(0, y-2):min(height, y+3), max(0, x-2):min(width, x+3)] = 1

        # Generate probabilistic prediction by blurring and adding noise
        preds[i] = gaussian_filter(masks[i].astype(np.float32), sigma=2)
        preds[i] += noise_level * rng.standard_normal((height, width))
        preds[i] = np.clip(preds[i], 0, 1)  # Keep probabilities in [0,1]

    if return_numpy:
        return preds, masks

    return torch.from_numpy(preds), torch.from_numpy(masks)
