import pytest
import torch
from pyaupro import PerRegionOverlap, auc_compute, generate_random_data
from numpy.testing import assert_approx_equal

preds, target = generate_random_data(seed=42)
preds_, target_ = generate_random_data(seed=1337)

def test_exact_computation():
    metric = PerRegionOverlap(thresholds=None)
    metric.update(preds, target)
    fpr, pro = metric.compute()
    assert isinstance(fpr, torch.Tensor)
    assert isinstance(pro, torch.Tensor)
    assert fpr.shape == pro.shape
    assert torch.all(fpr >= 0) and torch.all(fpr <= 1)
    assert torch.all(pro >= 0) and torch.all(pro <= 1)

@pytest.mark.parametrize("thresholds", [5, [0.2, 0.4, 0.6, 0.8]])
def test_approximated_computation(thresholds):
    metric = PerRegionOverlap(thresholds=thresholds)
    metric.update(preds, target)
    fpr, pro = metric.compute()

    assert isinstance(fpr, torch.Tensor)
    assert isinstance(pro, torch.Tensor)
    n_thresh = len(thresholds) if isinstance(thresholds, list) else thresholds
    assert fpr.shape == pro.shape == (n_thresh,)
    assert torch.all(fpr >= 0) and torch.all(fpr <= 1)
    assert torch.all(pro >= 0) and torch.all(pro <= 1)

def test_multiple_updates_exact():
    metric = PerRegionOverlap()
    metric.update(preds, target)
    metric.update(preds_, target_)
    fpr, pro = metric.compute()

    assert fpr.shape == pro.shape
    assert torch.all(fpr >= 0) and torch.all(fpr <= 1)
    assert torch.all(pro >= 0) and torch.all(pro <= 1)

def test_multiple_updates_approximate():
    metric = PerRegionOverlap(thresholds=10)
    metric.update(preds, target)
    metric.update(preds_, target_)
    fpr, pro = metric.compute()

    assert fpr.shape == pro.shape == (10,)
    assert torch.all(fpr >= 0) and torch.all(fpr <= 1)
    assert torch.all(pro >= 0) and torch.all(pro <= 1)

@pytest.mark.parametrize("data", [[(preds, target)], [(preds, target), ((preds_, target_))]])
def test_approx_auc_similar_to_exact_auc(data):
    metric_exact = PerRegionOverlap(thresholds=None)
    metric_approx = PerRegionOverlap(thresholds=1000)

    for (preds, target) in data:
        metric_approx.update(preds, target)
        metric_exact.update(preds, target)

    fpr_approx, pro_approx = metric_approx.compute()
    auc_approx = auc_compute(fpr_approx, pro_approx, reorder=True)
    fpr_exact, pro_exact = metric_exact.compute()
    auc_exact = auc_compute(fpr_exact, pro_exact, reorder=True)    

    # with a large number of thresholds (1000), the values should be close
    assert_approx_equal(auc_exact, auc_approx, significant=4)

def test_invalid_inputs():
    with pytest.raises(ValueError):
        PerRegionOverlap(thresholds=-1)

    with pytest.raises(ValueError):
        PerRegionOverlap(thresholds=1.5)

    with pytest.raises(ValueError):
        PerRegionOverlap(thresholds=[-0.1, 1.2])
