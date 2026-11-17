import runpy
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

RECIPES = Path(__file__).parents[1] / "recipes"


def run(recipe, client, argv, monkeypatch):
    monkeypatch.setitem(sys.modules, "nextron_cloud", NS(NexTron=lambda: client, Episode=NS))
    monkeypatch.setattr(sys, "argv", [recipe, *argv])
    runpy.run_path(str(RECIPES / recipe), run_name="__main__")


def test_a_changed_data_file_is_uploaded_as_a_new_dataset(tmp_path, monkeypatch):
    data = tmp_path / "prompts.jsonl"
    data.write_text('{"prompt": "1 + 1", "answer": "2"}\n')
    client = MagicMock()
    client.datasets.list.return_value = [NS(id="ds_old", name="math numeric (rft)", state="ready")]
    client.datasets.wait.return_value = NS(id="ds_new", samples=1)
    client.jobs.estimate.return_value = NS(credits=1)
    run("math_rft/train.py", client, ["--data", str(data)], monkeypatch)
    assert client.datasets.upload.call_args.args[0] == str(data)


def test_a_failed_job_ends_with_its_error(tmp_path, monkeypatch):
    for name in ("no_robots_train.jsonl", "no_robots_val.jsonl"):
        (tmp_path / name).write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    client = MagicMock()
    client.datasets.list.return_value = []
    client.datasets.wait.return_value = NS(id="ds_1", samples=100, tokens={"qwen3.8-27b": {"total": 100_000}})
    client.models.return_value = [NS(id="qwen3.8-27b", layouts={"1x8-h100": NS(gpus=8)})]
    client.jobs.estimate.return_value = NS(steps=5, credits=1, warnings=[])
    client.jobs.create.return_value = NS(id="job_1")
    client.jobs.wait.return_value = NS(state="failed", error="GPU initialization failed")
    with pytest.raises(SystemExit, match="GPU initialization failed"):
        run("chat_sft/train.py", client, ["--yes"], monkeypatch)


def test_a_failed_solution_gets_its_test_output_and_one_more_attempt(monkeypatch):
    task = NS(problem="Print the input.", tests=[{"input": "7\n", "output": "7\n"}])
    monkeypatch.setattr("tinker_cookbook.recipes.code_rl.code_env.load_deepcoder_tasks", lambda split: [task])

    async def check(tests, code):
        failed = {"output": "1\n", "expected": "7\n", "inputs": "7\n", "error_code": -2, "error_message": "Wrong Answer"}
        return "input" in code, {"run_result": {"stdout": f"{failed}\n"}}

    monkeypatch.setattr("tinker_cookbook.recipes.code_rl.code_grading.sandbox_check_correctness", check)
    wrong, right = NS(text="```python\nprint(1)\n```"), NS(text="```python\nprint(input())\n```")
    live = MagicMock()
    live.sample.side_effect = [[(wrong,)], [(right,)]]
    live.train.return_value = NS(reward_mean=1.0, zero_variance_fraction=1.0, truncated_fraction=0.0, generation_s=1.0,
                                 train_s=1.0, client_s=1.0, tokens=10, tokens_per_s_per_gpu=1.0, rows=1, row_fill=1.0,
                                 kl_sample_train=0.0)
    client = MagicMock()
    client.rl.connect.return_value.__enter__.return_value = live
    run("code_rl/train.py", client, ["rl_1", "--steps", "1", "--problems", "1", "--episodes", "1"], monkeypatch)
    told = live.sample.call_args_list[1].args[0][0][-1]["content"]
    assert "error_message: Wrong Answer" in told and "output: 1" in told
    assert "expected" not in told
    (episode,) = live.train.call_args.kwargs["episodes"]
    assert episode.samples == [wrong, right]
    assert episode.reward == 1.0


def test_a_one_step_job_still_reports_its_charge(tmp_path, monkeypatch, capsys):
    for name in ("no_robots_train.jsonl", "no_robots_val.jsonl"):
        (tmp_path / name).write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    client = MagicMock()
    client.datasets.list.return_value = []
    client.datasets.wait.return_value = NS(id="ds_1", samples=100, tokens={"qwen3.8-27b": {"total": 100_000}})
    client.models.return_value = [NS(id="qwen3.8-27b", layouts={"1x8-h100": NS(gpus=8)})]
    client.jobs.estimate.return_value = NS(steps=1, credits=1, warnings=[])
    client.jobs.create.return_value = NS(id="job_1")
    client.jobs.wait.return_value = NS(state="succeeded", error=None)
    client.jobs.metrics.return_value = [NS(step=1, total_steps=1, loss=1.0, val_loss=None, tokens_per_s_per_gpu=900)]
    client.jobs.billing.return_value = NS(charged_credits=3)
    run("chat_sft/train.py", client, ["--yes"], monkeypatch)
    assert "$0.03 charged" in capsys.readouterr().out


def code_rl(monkeypatch, check):
    """recipes/code_rl with a stand-in for the sandbox: `check(tests, code)` is its grader."""
    monkeypatch.setitem(sys.modules, "nextron_cloud", NS(NexTron=None, Episode=None))
    recipe = runpy.run_path(str(RECIPES / "code_rl/train.py"))
    monkeypatch.setitem(recipe["run_tests"].__globals__, "sandbox_check_correctness", check)
    return recipe


def test_a_sandbox_that_does_not_answer_stops_the_loop(monkeypatch):
    async def down(tests, code):
        return False, {"error": "connection refused"}

    recipe = code_rl(monkeypatch, down)
    live = MagicMock()
    live.sample.return_value = [(NS(text="```python\nprint(1)\n```"),)]
    with pytest.raises(RuntimeError, match="connection refused"):
        recipe["play"](live, [NS(problem="p", tests=[])], 1, 16)


def test_no_second_attempt_is_sampled_when_every_first_one_passes(monkeypatch):
    async def passes(tests, code):
        return True, {"run_result": {"stdout": ""}}

    recipe = code_rl(monkeypatch, passes)
    live = MagicMock()
    live.sample.return_value = [(NS(text="```python\nprint(1)\n```"),)]
    games = recipe["play"](live, [NS(problem="p", tests=[])], 1, 16)
    assert live.sample.call_count == 1 and games[0]["reward"] == 1
