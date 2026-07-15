import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _tree(relative_path):
    return ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))


def test_yolo26_does_not_call_vendor_trainer_or_validator():
    paths = [
        "hq_det/tools/train_yolo26.py",
        "scripts/run_evaluate_yolo26.py",
    ]
    forbidden_modules = {
        "ultralytics.engine.trainer",
        "ultralytics.models.yolo.detect.train",
        "ultralytics.models.yolo.detect.val",
    }
    forbidden_names = {"BaseTrainer", "DetectionTrainer", "DetectionValidator"}

    for path in paths:
        tree = _tree(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in forbidden_modules
                assert forbidden_names.isdisjoint(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                assert forbidden_modules.isdisjoint(alias.name for alias in node.names)


def test_yolo26_uses_framework_trainer_and_industrial_validation():
    tree = _tree("hq_det/tools/train_yolo26.py")
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Yolo26Trainer"
    )
    assert [base.id for base in trainer.bases if isinstance(base, ast.Name)] == [
        "YoloTrainer"
    ]
    called_attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "valid_epoch" in called_attributes


def test_frozen_config_does_not_import_default_cfg():
    tree = _tree("hq_det/training/recipes/ultralytics_v847/config.py")
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "DEFAULT_CFG" not in imported_names


def test_build_argument_calls_match_the_helper_signature():
    tree = _tree("hq_det/tools/train_yolo26.py")
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_arguments"
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_build_arguments"
    ]
    assert calls
    assert all(len(call.args) == len(helper.args.args) for call in calls)


def test_dependency_is_pinned_to_847():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"ultralytics==8.4.7"' in pyproject


def test_recipe_exposes_all_framework_component_boundaries():
    tree = _tree("hq_det/training/recipes/ultralytics_v847/recipe.py")
    recipe = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UltralyticsV847Recipe"
    )
    assigned = {
        target.attr
        for node in ast.walk(recipe)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    assert {
        "augmentation",
        "dataloader",
        "ema",
        "loss",
        "optimizer",
        "postprocess",
        "precision",
        "runtime",
        "scheduler",
    }.issubset(assigned)
