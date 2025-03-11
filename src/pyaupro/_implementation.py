from typing import Any, List, Optional, Union

import torch
import numpy as np
from scipy.ndimage import label
from torch import Tensor

from torchmetrics.functional.classification.precision_recall_curve import (
    _adjust_threshold_arg,
    _binary_precision_recall_curve_arg_validation,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_curve

from ._reference import compute_pro
from ._utilities import auc_compute

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = [
        "PerRegionOverlap.plot",
    ]


class PerRegionOverlap(Metric):
    r"""Compute the per-region overlap curve for binary tasks.

    The curve consist of multiple pairs of region-overlap and false-positive rate values evaluated at different
    thresholds, such that the tradeoff between the two values can been seen.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, ...)``. Preds should be a tensor containing
      probabilities or logits for each observation. If preds has values outside [0,1] range we consider the input
      to be logits and will auto apply sigmoid per element.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``. Target should be a tensor containing
      ground truth labels, and therefore only contain {0,1} values (except if `ignore_index` is specified). The value
      1 always encodes the positive class.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``per-region overlap`` (:class:`~torch.Tensor`): if `thresholds=None` a list for each class is returned with an 1d
      tensor of size ``(n_thresholds+1, )`` with precision values (length may differ between classes). If `thresholds`
      is set to something else, then a single 2d tensor of size ``(n_classes, n_thresholds+1)`` with precision values
      is returned.
    - ``recall`` (:class:`~torch.Tensor`): if `thresholds=None` a list for each class is returned with an 1d tensor
      of size ``(n_thresholds+1, )`` with recall values (length may differ between classes). If `thresholds` is set to
      something else, then a single 2d tensor of size ``(n_classes, n_thresholds+1)`` with recall values is returned.
    - ``thresholds`` (:class:`~torch.Tensor`): if `thresholds=None` a list for each class is returned with an 1d
      tensor of size ``(n_thresholds, )`` with increasing threshold values (length may differ between classes). If
      `threshold` is set to something else, then a single 1d tensor of size ``(n_thresholds, )`` is returned with
      shared threshold values for all classes.

    .. note::
       The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
       that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
       non-binned version that uses memory of size :math:`\mathcal{O}(n_{samples})` whereas setting the `thresholds`
       argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
       size :math:`\mathcal{O}(n_{thresholds})` (constant memory).

    Arguments:
        thresholds:
            Can be one of:
            - If set to `None`, will use a non-binned reference approach provided by the authors of MVTecAD, where
                no thresholds are explicitly calculated. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
                0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
                bins for the calculation.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from pyaupro import PerRegionOverlap
        >>> preds = torch.tensor([0, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> bprc = PerRegionOverlap(thresholds=None)
        >>> bprc(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        >>> (tensor([0.5000, 0.6667, 0.5000, 0.0000, 1.0000]),
        >>> tensor([1.0000, 1.0000, 0.5000, 0.0000, 0.0000]),
        >>> tensor([0.0000, 0.5000, 0.7000, 0.8000]))
        >>> bprc = PerRegionOverlap(thresholds=5)
        >>> bprc(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        >>> (tensor([0.5000, 0.6667, 0.6667, 0.0000, 0.0000, 1.0000]),
        >>>  tensor([1., 1., 1., 0., 0., 0.]),
        >>>  tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))
    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    preds: List[Tensor]
    target: List[Tensor]
    fpr: Tensor
    pro: Tensor
    num_updates: Tensor

    def __init__(
        self,
        thresholds: Optional[Union[int, list[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)

        self.ignore_index = ignore_index
        self.validate_args = validate_args

        thresholds = _adjust_threshold_arg(thresholds)
        if thresholds is None:
            self.thresholds = thresholds
            self.add_state("preds", default=[], dist_reduce_fx="cat")
            self.add_state("target", default=[], dist_reduce_fx="cat")
        else:
            self.register_buffer("thresholds", thresholds, persistent=False)
            self.add_state("fpr", default=torch.zeros(len(thresholds), dtype=torch.double), dist_reduce_fx="mean")
            self.add_state("pro", default=torch.zeros(len(thresholds), dtype=torch.double), dist_reduce_fx="mean")
            self.add_state("num_updates", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states."""
        if self.thresholds is None:
            self.preds.append(preds)
            self.target.append(target)
        else:
            fpr, pro = _per_region_overlap_update(preds, target, self.thresholds)
            self.fpr += fpr
            self.pro += pro
            self.num_updates += 1

    def compute(self) -> tuple[Tensor, Tensor]:
        """Compute metric."""
        if self.thresholds is None:
            # calculate the exact fpr and pro using all distinct thresholds using the logic
            # from torchmetrics ``_binary_precision_recall_curve_compute`` and ``_binary_clf_curve``
            preds, target = dim_zero_cat(self.preds), dim_zero_cat(self.target)
            fpr, pro = compute_pro(preds.numpy(force=True), target.numpy(force=True))
            return torch.from_numpy(fpr), torch.from_numpy(pro)

        return self.fpr / self.num_updates, self.pro / self.num_updates

    def plot(
        self,
        curve: Optional[tuple[Tensor, Tensor]] = None,
        score: Optional[Union[Tensor, bool]] = None,
        ax: Optional[_AX_TYPE] = None,
        limit: float = 1.0,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            curve: The output of either `metric.compute` or `metric.forward`. If no value is provided, will
                automatically call `metric.compute` and plot that result.
            score: Provide a area-under-the-curve score to be displayed on the plot. If `True` and no curve is provided,
                will automatically compute the score. The score is computed by using the trapezoidal rule to compute the
                area under the curve.
            ax: An matplotlib axis object. If provided will add plot to that axis
            limit: Integration limit chosen for the FPR such that only the values until the limit are computed / plotted.

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from torch import rand, randint
            >>> from torchmetrics.classification import BinaryROC
            >>> preds = rand(20)
            >>> target = randint(2, (20,))
            >>> metric = BinaryROC()
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot(score=True)

        """
        fpr, pro = curve or self.compute()
        score, fpr, pro = auc_compute(
            fpr,
            pro,
            limit=limit,
            reorder=True,
            return_curve=True
        )
        return plot_curve(
            (fpr, pro),
            score=score,
            ax=ax,
            label_names=("False positive rate", "Per-region overlap"),
            name=self.__class__.__name__,
        )


def _per_region_overlap_update(
    preds: Tensor,
    target: Tensor,
    thresholds:  Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the false positive rate and per-region overlap.

    This implementation loops over thresholds for memory efficiency, but vectorization should be possible as well.
    """
    len_t = len(thresholds)
    negatives = target == 0
    total_negatives = negatives.sum()
    false_positive_rate = thresholds.new_empty(len_t, dtype=torch.float64)
    per_region_overlap = thresholds.new_empty(len_t, dtype=torch.float64)
    batch_structure = np.zeros((3,3,3), dtype=int)
    batch_structure[1,:,:] = 1

    # pre-compute total component areas for region overlap
    components, _ = label(target.numpy(force=True), structure=batch_structure)
    flat_components = torch.from_numpy(components.ravel())
    # only keep true components (non-zero values)
    pos_comp_mask = flat_components > 0
    flat_components = flat_components[pos_comp_mask]
    total_area = torch.bincount(flat_components)[1:]

    # Iterate one threshold at a time to conserve memory
    for i in range(len_t):
        # compute false positive rate
        preds_t = preds >= thresholds[i]
        false_positive_rate[i] = negatives[preds_t].sum() / total_negatives

        # compute per-region overlap
        overlap_area = torch.bincount(
            flat_components,
            weights=preds_t.ravel()[pos_comp_mask]
        )[1:]
        per_region_overlap[i] = torch.mean(overlap_area / total_area)

    return false_positive_rate, per_region_overlap
