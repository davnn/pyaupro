## ``pyaupro``: Efficient Per-Region Overlap Computation

This package is intended to compute the per-region overlap metric using an
efficient [torchmetrics](https://github.com/Lightning-AI/torchmetrics) implementation.

If you are used to ``torchmetrics``, for example to ``BinaryROC``, you will find
yourself at home using ``pyaupro``.

We export a single metric called ``PerRegionOverlap``, which is described in the paper
referenced below.

    Bergmann, Paul, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger. “The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.” International Journal of Computer Vision 129, no. 4 (April 1, 2021): 1038–59. https://doi.org/10.1007/s11263-020-01400-4.

The arguments to instantiate the metric are as follows.

```
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
```

An ``update`` of the metric expects a three dimensional ``preds`` tensor where the first dimension is the batch dimension (floats between zero and one, otherwise the values are considered logits) equally shaped ``target`` tensor containing ground truth labels, and therefore only contain {0,1} values.

If ``thresholds`` is ``None``, the metric computes an exact Per-Region Overlap (PRO) curve over all possible values. In this case, each update step appends the given tensors and the calculation happens at ``compute``. We use the official implementation provided in ``MVTecAD`` as a reference.

If thresholds are given, the computation is approximate and happens at each update step. In the approximate case, ``compute`` returns a mean of the batched computations during update.

We further provide an ``auc_compute`` utility for area under the curve computation, which is also used
in ``PerRegionOverlap`` if ``score=True``. The arguments for ``auc_compute`` are as follows.

```
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
```

### How to develop

- Use ``uv sync`` to install dependencies from the lock file.
- Use ``uv lock`` to update the lock file given the pinned dependencies.
- Use ``uv lock --upgrade`` to upgrade the lock file ignoring pinned dependencies.
- Use ``uv pip install --editable .`` to install the local package.
- Use ``uv run pytest tests`` to test the local package.

It might happen that the host ``github.com`` is not trusted, in this case use ``uv sync --allow-insecure-host https://github.com`` if you trust ``github.com``.
