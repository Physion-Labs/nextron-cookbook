import importlib
import sys
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

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


def rl_run(monkeypatch, model):
    """Point the CLI at a stand-in for an RL run that trains `model`."""
    monkeypatch.setenv("TINKER_BASE_URL", "https://api.test/v1/training/sessions/rl_1/tinker")
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    tinker = MagicMock()
    tinker.RestClient.return_value.get_server_capabilities.return_value.result.return_value.supported_models = [
        NS(model_name=model, max_context_length=16384)]
    monkeypatch.setitem(sys.modules, "tinker", tinker)


def test_run_starts_a_recipe_only_on_an_rl_run_of_its_model(monkeypatch):
    started = []

    def call(cmd):
        started.append(cmd)
        return 0

    monkeypatch.setattr(cli.subprocess, "call", call)
    rl_run(monkeypatch, "google/gemma-4-26B-A4B-it")
    with pytest.raises(SystemExit, match="trains google/gemma-4-26B-A4B-it.*tested on Qwen/Qwen3.8-27B"):
        cli.main(["run", "gsm8k"])
    assert not started
    rl_run(monkeypatch, "Qwen/Qwen3.8-27B")
    with pytest.raises(SystemExit) as ended:
        cli.main(["run", "gsm8k"])
    assert ended.value.code == 0 and len(started) == 1


def test_check_says_when_the_runs_tokenizer_cannot_be_loaded(monkeypatch):
    def missing(name):
        raise OSError(f"{name} is not a valid model identifier")

    rl_run(monkeypatch, "Qwen/Qwen3.8-Flash-Next-180B")
    monkeypatch.setattr("tinker_cookbook.tokenizer_utils.get_tokenizer", missing)
    with pytest.raises(SystemExit, match="tokenizer could not be loaded.*Qwen3.8-Flash-Next-180B is not a valid model identifier"):
        cli.main(["check"])


def test_sl_renders_its_data_with_thinking_off_and_follows_a_renderer_override():
    from tinker_cookbook.recipes import sl_basic

    def data_renderer(overrides):
        blueprint = sl_basic.build_config_blueprint("Qwen/Qwen3.8-27B")
        config = blueprint.make_from_argv(arguments("sl", overrides))
        return config.dataset_builder.common_config.renderer_name

    assert data_renderer([]) == "qwen3_8_disable_thinking"
    assert data_renderer(["renderer_name=qwen3_8_low_reasoning"]) == "qwen3_8_low_reasoning"
