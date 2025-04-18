from typing import Any
from warnings import warn

import numpy as np
import torch
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
        changepoints_only:
            Modify the exact curve to retain the relevant points only.
        reference_implementation:
            Fall back to the official MVTecAD implementation for the exact computation.
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.

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
    higher_is_better: bool | None = None
    full_state_update: bool = False

    preds: list[Tensor]
    target: list[Tensor]
    fpr: Tensor
    pro: Tensor
    num_updates: Tensor

    def __init__(
        self,
        thresholds: int | list[float] | Tensor | None = None,
        ignore_index: int | None = None,
        *,
        validate_args: bool = True,
        changepoints_only: bool = True,
        reference_implementation: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)

        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.reference_implementation = reference_implementation
        self.changepoints_only = changepoints_only

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
        assert preds.shape == target.shape, "Cannot update when preds.shape != target.shape."

        if self.thresholds is None:
            self.preds.append(preds)
            self.target.append(target)
            return

        # compute the fpr and pro for all preds and target given thresholds
        if (res := _per_region_overlap_update(preds, target, self.thresholds)) is not None:
            fpr, pro = res
            # weight the fpr and pro contribution given the number of samples
            num_samples = len(preds)
            self.fpr += fpr * num_samples
            self.pro += pro * num_samples
            self.num_updates += num_samples

    def compute(self) -> tuple[Tensor, Tensor]:
        """Compute metric."""
        if self.thresholds is None:
            # calculate the exact fpr and pro using all distinct thresholds using the logic
            # from torchmetrics ``_binary_precision_recall_curve_compute`` and ``_binary_clf_curve``
            preds, target = dim_zero_cat(self.preds), dim_zero_cat(self.target)
            assert preds.shape == target.shape, "Cannot compute PRO when preds.shape != target.shape."

            if self.reference_implementation:
                assert preds.ndim == 3, "The reference implementation is only defined for 3d-tensors."
                fpr, pro = compute_pro(preds.numpy(force=True), target.numpy(force=True))
                return torch.from_numpy(fpr), torch.from_numpy(pro)

            return _per_region_overlap_compute(preds, target, changepoints_only=self.changepoints_only)

        return self.fpr / self.num_updates, self.pro / self.num_updates

    def plot(
        self,
        curve: tuple[Tensor, Tensor] | None = None,
        score: Tensor | bool | None = None,
        ax: _AX_TYPE | None = None,
        limit: float = 1.0,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            curve: The output of either `metric.compute` or `metric.forward`. If no value is provided, will
                automatically call `metric.compute` and plot that result.
            score: Provide a area-under-the-curve score to be displayed on the plot. If `True` and no curve is provided,
                will automatically compute the score. The score is computed by using the trapezoidal rule to compute the
                area under the curve.
            ax:
                An matplotlib axis object. If provided will add plot to that axis
            limit:
                Integration limit chosen for the FPR such that only the values until the limit are computed / plotted.

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
            descending=True,
            return_curve=True,
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
    thresholds: Tensor,
) -> tuple[Tensor, Tensor] | None:
    """Return the false positive rate and per-region overlap for the given thresholds."""
    # pre-compute total component areas for region overlap
    structure = _get_structure(target.ndim)
    components, n_components = label(target.numpy(force=True), structure=structure)

    # ensure that there are components available for overlap calculation
    if n_components == 0:
        return warn("No regions found in target for update, ignoring update.", stacklevel=2)

    # convert back to torch and flatten components to vector
    flat_components = torch.from_numpy(components.ravel())

    # only keep true components (non-zero values)
    pos_comp_mask = flat_components > 0
    flat_components = flat_components[pos_comp_mask]
    region_total_area = torch.bincount(flat_components)[1:]

    # pre-compute the negative mask and flatten preds for perf
    negatives = (target == 0).ravel()
    total_negatives = negatives.sum()
    flat_preds = preds.ravel()

    # initialize the result tensors
    len_t = len(thresholds)
    false_positive_rate = thresholds.new_empty(len_t, dtype=torch.float64)
    per_region_overlap = thresholds.new_empty(len_t, dtype=torch.float64)

    # Iterate one threshold at a time to conserve memory
    for t in range(len_t):
        # compute false positive rate
        preds_t = flat_preds >= thresholds[t]
        false_positive_rate[t] = negatives[preds_t].sum()

        # compute per-region overlap
        region_overlap_area = torch.bincount(
            flat_components,
            weights=preds_t[pos_comp_mask],
            minlength=n_components,
        )[1:]
        # faster than region_overlap_area / region_total_area
        region_overlap_area.div_(region_total_area)
        per_region_overlap[t] = torch.mean(region_overlap_area)

    return false_positive_rate / total_negatives, per_region_overlap


def _per_region_overlap_compute(
    preds: Tensor,
    target: Tensor,
    *,
    changepoints_only: bool,
) -> tuple[Tensor, Tensor]:
    """Compute the exact per-region overlap over all possible thresholds."""
    # Structuring element for computing connected components.
    structure = _get_structure(target.ndim)

    # pre-compute total component areas for region overlap
    components, n_components = label(target.numpy(force=True), structure=structure)
    assert n_components > 0, "Cannot compute PRO metric without regions, found 0 regions in target."

    # convert back to torch, type cast necessary for later .take call
    flat_components = torch.from_numpy(components.ravel()).type(torch.LongTensor)

    # contribution of per-region overlap to the curve
    # only use real components for bincount (nonzero values)
    # divide the relative contribution by area and number of components
    bin_contribution = torch.bincount(flat_components[flat_components > 0], minlength=n_components).to(torch.float64)
    bin_contribution.reciprocal_().divide_(n_components)
    bin_contribution[0] = 0.0

    # sort the contributions according to predictions
    sort_idx = _fast_argsort(preds)
    flat_components = flat_components.take(sort_idx)

    # contribution of false positive rate to the curve
    false_positive_rate = (flat_components == 0).to(torch.float64)
    false_positive_rate.div_(false_positive_rate.sum())
    torch.cumsum(false_positive_rate, dim=0, out=false_positive_rate)

    # contribution of the bin to the curve
    per_region_overlap = bin_contribution.take(flat_components)
    torch.cumsum(per_region_overlap, dim=0, out=per_region_overlap)

    # prevent possible cumsum rounding errors
    torch.clamp_(false_positive_rate, max=1.0)
    torch.clamp_(per_region_overlap, max=1.0)

    if changepoints_only:
        mask = _changepoint_mask(false_positive_rate, per_region_overlap)
        return false_positive_rate[mask], per_region_overlap[mask]

    return false_positive_rate, per_region_overlap


def _changepoint_mask(fpr: Tensor, pro: Tensor) -> tuple[Tensor, Tensor]:
    """Filter the curve values to retain only the relevant points."""
    const_pro = pro[:-1] == pro[1:]
    const_fpr = fpr[:-1] == fpr[1:]
    mask_pro = const_pro[:-1].logical_and(const_pro[1:])
    mask_fpr = const_fpr[:-1].logical_and(const_fpr[1:])
    mask_pro.logical_or_(mask_fpr).logical_not_()
    return torch.cat([torch.BoolTensor([True]), mask_pro, torch.BoolTensor([True])])


def _fast_argsort(values: Tensor) -> Tensor:
    """Sort using numpy, it seems to be much faster than pytorch here."""
    np_idx = np.argsort(-values.ravel().numpy(force=True))
    return torch.from_numpy(np_idx)


def _get_structure(ndim: int) -> np.ndarray:
    """Define a batched 2-neighborhood (2d) and 8-neighborhood (3d) structuring element."""
    assert 1 < ndim < 4, f"Cannot compute PRO for tensors with < 2 or > 3 dims, found {ndim}."
    structure = np.zeros((3,) * ndim, dtype=bool)
    structure[1] = True
    return structure
