import pytest
import torch
from pyaupro import PerRegionOverlap, auc_compute, generate_random_data
from numpy.testing import assert_approx_equal

# generate samples of unequal batch size to test if the updates are correctly
# weighted and vary ``noise_level`` to generate batches of differing AUC.
# additionally vary the number of objects for overlap calculation.
noise_levels = [0.2, 0.4, 0.6]
d0, d1, d2 = [generate_random_data(
    seed=i, batch_size=i + 1, num_objects=i + 2, noise_level=n
) for i, n in enumerate(noise_levels)]
test_data = [[d0], [d1, d2], [d0, d1, d2]]

@pytest.mark.parametrize("data", test_data)
def test_exact_computation(data):
    metric = PerRegionOverlap(thresholds=None)
    for (preds, target) in data:
        metric.update(preds, target)

    fpr, pro = metric.compute()
    assert isinstance(fpr, torch.Tensor)
    assert isinstance(pro, torch.Tensor)
    assert fpr.shape == pro.shape
    assert torch.all(fpr >= 0) and torch.all(fpr <= 1)
    assert torch.all(pro >= 0) and torch.all(pro <= 1)

@pytest.mark.parametrize("data", test_data)
@pytest.mark.parametrize("thresholds", [5, 10, [0.2, 0.4, 0.6, 0.8]])
def test_approximated_computation(data, thresholds):
    metric = PerRegionOverlap(thresholds=thresholds)
    for (preds, target) in data:
        metric.update(preds, target)

    fpr, pro = metric.compute()
    assert isinstance(fpr, torch.Tensor)
    assert isinstance(pro, torch.Tensor)
    n_thresh = len(thresholds) if isinstance(thresholds, list) else thresholds
    assert fpr.shape == pro.shape == (n_thresh,)
    assert torch.all(fpr >= 0) and torch.all(fpr <= 1)
    assert torch.all(pro >= 0) and torch.all(pro <= 1)

@pytest.mark.parametrize("data", test_data)
@pytest.mark.parametrize("thresholds", [100, torch.linspace(0, 1, 100)])
def test_approx_auc_similar_to_exact_auc(data, thresholds):
    metric_exact = PerRegionOverlap(thresholds=None)
    metric_approx = PerRegionOverlap(thresholds=thresholds)

    for (preds, target) in data:
        metric_approx.update(preds, target)
        metric_exact.update(preds, target)

    fpr_approx, pro_approx = metric_approx.compute()
    auc_approx = auc_compute(fpr_approx, pro_approx, reorder=True)
    fpr_exact, pro_exact = metric_exact.compute()
    auc_exact = auc_compute(fpr_exact, pro_exact, reorder=True)    

    # with a large enough number of thresholds, the values should be close
    assert_approx_equal(auc_exact, auc_approx, significant=2)

def test_invalid_inputs():
    with pytest.raises(ValueError):
        PerRegionOverlap(thresholds=-1)

    with pytest.raises(ValueError):
        PerRegionOverlap(thresholds=1.5)

    with pytest.raises(ValueError):
        PerRegionOverlap(thresholds=[-0.1, 1.2])
