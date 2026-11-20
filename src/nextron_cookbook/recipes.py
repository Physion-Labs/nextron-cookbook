"""tinker-cookbook recipes with the settings they were measured with on NexCloud (docs/results.md).

A recipe here is the cookbook's own module plus `key=value` arguments. Nothing is copied from the cookbook: `run`
starts `python -m <module>` with these arguments, and any `key=value` you add replaces ours.
"""
from dataclasses import dataclass

MODEL = "Qwen/Qwen3.8-27B"
# Full-parameter training: the learning rate is about 10x lower than the cookbook's LoRA defaults.
COMMON = {"model_name": MODEL, "renderer_name": "qwen3_8_disable_thinking", "save_every": "0"}
# The cookbook's default loss for these recipes is stable_reinforce, which NexCloud does not have yet.
RL = {"loss_fn": "importance_sampling", "learning_rate": "2e-6", "behavior_if_log_dir_exists": "delete"}


@dataclass(frozen=True)
class Recipe:
    module: str
    summary: str
    args: dict
    needs: str = ""           # what must be set up first
    min_context: int = 9216   # the RL run's max context this recipe needs
    extra: str = ""           # pip extra of nextron-cookbook that brings the recipe's dependencies


RECIPES = {
    "gsm8k": Recipe(
        "tinker_cookbook.recipes.math_rl.train",
        "Grade-school math, 3 short steps: the quickest check that an RL run trains end to end (about 2 minutes).",
        {**COMMON, **RL, "env": "gsm8k", "group_size": "4", "groups_per_batch": "16", "max_tokens": "1024",
         "learning_rate": "1e-6", "max_steps": "3", "eval_every": "0"}, extra="math"),
    "math": Recipe(
        "tinker_cookbook.recipes.math_rl.train",
        "Hendrycks MATH with a 512-token reply budget; held-out MATH-500 before and after.",
        {**COMMON, **RL, "env": "math", "group_size": "8", "groups_per_batch": "64", "max_tokens": "512",
         "max_steps": "20", "eval_every": "10"}, extra="math"),
    "code": Recipe(
        "tinker_cookbook.recipes.code_rl.train",
        "Competitive programming (DeepCoder): two turns, a tool call, generated code runs in SandboxFusion.",
        {**COMMON, **RL, "group_size": "8", "groups_per_batch": "32", "max_tokens": "4096", "max_steps": "10",
         "eval_every": "10"},
        needs="a SandboxFusion server and SANDBOX_URL (docs/recipes.md)", min_context=16384),
    "sl": Recipe(
        "tinker_cookbook.recipes.sl_basic",
        "Supervised fine-tuning on chat data (no_robots) through an RL run: loss_fn cross_entropy.",
        # sl_basic renders its data with the model's default renderer (thinking on) unless the dataset is given here.
        {**COMMON, "learning_rate": "1e-5", "eval_every": "0",
         "dataset_builder": "tinker_cookbook.recipes.chat_sl.chat_datasets:NoRobotsBuilder",
         "dataset_builder.common_config.max_length": "32768", "dataset_builder.common_config.batch_size": "128"}),
}


def arguments(name, overrides):
    """The recipe's arguments with `overrides` (a list of key=value) applied on top, as a list of key=value."""
    args = dict(RECIPES[name].args)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {item!r}")
        args[key] = value
    if "dataset_builder" in args:
        args.setdefault("dataset_builder.common_config.model_name_for_tokenizer", args["model_name"])
        args.setdefault("dataset_builder.common_config.renderer_name", args["renderer_name"])
    return [f"{key}={value}" for key, value in args.items()]
