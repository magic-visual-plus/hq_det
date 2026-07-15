from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from hq_det.training import OptimizerContext
from hq_det.training.recipes.ultralytics_v847.musgd import MuSGD
from hq_det.training.recipes.ultralytics_v847.optimizer import (
    UltralyticsV847Optimizer,
)
from hq_det.training.recipes.ultralytics_v847.schedule import (
    UltralyticsV847Scheduler,
)


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cv3 = torch.nn.Conv2d(4, 4, 1)


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Conv2d(3, 4, 3)
        self.bn = torch.nn.BatchNorm2d(4)
        self.model = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(23)] + [_Head()]
        )
        self.scalar_weight = torch.nn.Parameter(torch.ones(4))


def _hyp(optimizer="auto"):
    return SimpleNamespace(
        optimizer=optimizer,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        nbs=64,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=False,
    )


def test_auto_builds_exact_short_run_musgd_recipe():
    model = _ToyModel()
    hyp = _hyp()
    context = OptimizerContext(
        num_classes=15,
        dataset_size=2820,
        epochs=135,
        global_batch_size=4,
        nominal_batch_size=64,
    )
    result = UltralyticsV847Optimizer().build(model, hyp, context)

    assert isinstance(result.optimizer, MuSGD)
    assert result.name == "MuSGD"
    assert result.learning_rate == round(0.002 * 5 / (4 + 15), 6)
    assert result.momentum == 0.9
    assert result.accumulate == 16
    assert result.scaled_weight_decay == pytest.approx(0.0005)
    assert result.iterations == 6075
    assert result.optimizer.muon == 0.5
    assert result.optimizer.sgd == 0.5
    assert hyp.warmup_bias_lr == 0.0
    assert len(result.optimizer.param_groups) == 8

    head_weight = model.model[23].cv3.weight
    head_group = next(
        group
        for group in result.optimizer.param_groups
        if any(parameter is head_weight for parameter in group["params"])
    )
    assert head_group["param_group"] == "muon"
    assert head_group["lr"] == pytest.approx(result.learning_rate * 3)


def test_long_run_uses_847_muon_sgd_factors():
    result = UltralyticsV847Optimizer().build(
        _ToyModel(),
        _hyp(),
        OptimizerContext(
            num_classes=15,
            dataset_size=10000,
            epochs=100,
            global_batch_size=4,
            nominal_batch_size=64,
        ),
    )
    assert result.iterations > 10000
    assert result.learning_rate == 0.01
    assert result.optimizer.muon == 0.1
    assert result.optimizer.sgd == 1.0


def test_scheduler_and_warmup_use_847_batch_formulas():
    hyp = _hyp(optimizer="MuSGD")
    optimizer_result = UltralyticsV847Optimizer().build(
        _ToyModel(),
        hyp,
        OptimizerContext(
            num_classes=15,
            dataset_size=640,
            epochs=100,
            global_batch_size=4,
            nominal_batch_size=64,
        ),
    )
    strategy = UltralyticsV847Scheduler()
    schedule = strategy.build(optimizer_result.optimizer, hyp, epochs=100)

    assert schedule.lr_lambda(0) == pytest.approx(1.0)
    assert schedule.lr_lambda(100) == pytest.approx(0.01)
    accumulate = strategy.apply_warmup(
        optimizer_result.optimizer,
        hyp,
        schedule.lr_lambda,
        iteration=0,
        epoch=0,
        batches_per_epoch=160,
        base_accumulate=16,
        global_batch_size=4,
    )
    assert accumulate == 1
    bias_groups = [
        group
        for group in optimizer_result.optimizer.param_groups
        if group.get("param_group") == "bias"
    ]
    assert bias_groups
    assert all(group["lr"] == pytest.approx(0.1) for group in bias_groups)
