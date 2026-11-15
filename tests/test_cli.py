import importlib

import pytest

from nextron_cookbook import cli
from nextron_cookbook.recipes import RECIPES, arguments


def test_overrides_replace_tested_settings():
    args = arguments("math", ["max_steps=5", "wandb_project=mine"])
    assert "max_steps=5" in args and "max_steps=20" not in args
    assert "wandb_project=mine" in args
    assert "loss_fn=importance_sampling" in args


def test_every_recipe_is_a_cookbook_module():
    for recipe in RECIPES.values():
        importlib.import_module(recipe.module)


def test_run_needs_an_rl_run(monkeypatch):
    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    with pytest.raises(SystemExit, match="TINKER_BASE_URL"):
        cli.main(["run", "gsm8k"])
