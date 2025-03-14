## ``pyaupro``: Efficient Per-Region Overlap Computation

This package is intended to compute the per-region overlap metric using an
efficient [torchmetrics](https://github.com/Lightning-AI/torchmetrics) implementation.

If you are used to ``torchmetrics``, for example to ``BinaryROC``, you will find
yourself at home using ``pyaupro``.

We export a single metric called ``PerRegionOverlap``, which is described in the paper
referenced below.

    Bergmann, Paul, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger. “The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.” International Journal of Computer Vision 129, no. 4 (April 1, 2021): 1038–59. https://doi.org/10.1007/s11263-020-01400-4.

### Usage Example

```python
from pyaupro import PerRegionOverlap, auc_compute, generate_random_data

# generate random data for testing
preds, target = generate_random_data(batch_size=1, seed=42)

# initialize an approximate PRO-metric with 100 thresholds
pro_curve = PerRegionOverlap(thresholds=100)

# update the metric with the random preds and target
pro_curve.update(preds, target)

# compute the fpr and pro values for the curve
fpr, pro = pro_curve.compute()

# calculate the area under the curve
score = auc_compute(fpr, pro, reorder=True)

# plot the curve
pro_curve.plot(score=True)
```

### Usage Details

The arguments to instantiate ``pyaupro.PerRegionOverlap`` are as follows.

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
changepoints_only:
    Modify the exact curve to retain the relevant points only.
reference_implementation:
    Fall back to the official MVTecAD implementation for the exact computation.
kwargs:
    Additional keyword arguments, see :ref:`Metric kwargs` for more info.
```

An ``update`` of the metric expects a three-dimensional ``preds`` tensor where the first dimension is the batch dimension (floats between zero and one; otherwise, the values are considered logits) and an equally shaped ``target`` tensor containing binary ground truth labels ({0,1} values).

If ``thresholds`` is ``None``, the metric computes an exact Per-Region Overlap (PRO) curve over all possible values. In this case, each update step appends the given tensors, and the calculation happens in ``compute``. We use the official implementation provided in ``MVTecAD`` for exact calculation.

If thresholds are given, the computation is approximate and happens at each update step. In the approximate case, ``compute`` returns a mean of the batched computations during update.

We provide an ``auc_compute`` utility for area under the curve computation, which is also used
in ``PerRegionOverlap.plot`` if ``score=True``. The arguments for ``pyaupro.auc_compute`` are as follows.

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
- Use ``uv lock`` to update the lock file if necessary given the pinned dependencies.
- Use ``uv lock --upgrade`` to upgrade the lock file the latest valid dependencies.
- Use ``uv pip install --editable .`` to install the local package.
- Use ``uv run pytest tests`` to test the local package.

It might happen that the host ``github.com`` is not trusted, in this case use ``uv sync --allow-insecure-host https://github.com`` if you trust ``github.com``.
