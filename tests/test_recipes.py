import runpy
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

RECIPES = Path(__file__).parents[1] / "recipes"
STEP = NS(reward_mean=1.0, zero_variance_fraction=1.0, truncated_fraction=0.0, generation_s=1.0, train_s=1.0, client_s=1.0,
          tokens=10, tokens_per_s_per_gpu=1.0, rows=1, row_fill=1.0, kl_sample_train=0.0)   # what live.train returns


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
    live.train.return_value = STEP
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


@pytest.mark.parametrize("steps, line", [(21, "step    6 of 21  val_loss 1.7200"), (None, "step    6  val_loss 1.7200")])
def test_chat_sft_prints_each_validation_loss_while_the_job_trains(steps, line, tmp_path, monkeypatch, capsys):
    for name in ("no_robots_train.jsonl", "no_robots_val.jsonl"):
        (tmp_path / name).write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    printed_while_training = []

    def wait(job_id):
        printed_while_training.append(capsys.readouterr().out)
        return NS(state="succeeded", error=None)

    client = MagicMock()
    client.datasets.list.return_value = []
    client.datasets.wait.return_value = NS(id="ds_1", samples=100, tokens={"qwen3.8-27b": {"total": 100_000}})
    client.models.return_value = [NS(id="qwen3.8-27b", layouts={"1x8-h100": NS(gpus=8)})]
    client.jobs.estimate.return_value = NS(steps=steps, credits=1, warnings=[])
    client.jobs.create.return_value = NS(id="job_1")
    client.jobs.events.return_value = [NS(type="training.step", data={"step": 6, "total_steps": 21, "loss": 1.9}),
                                       NS(type="training.validation", data={"step": 6, "val_loss": 1.72})]
    client.jobs.wait.side_effect = wait
    client.jobs.metrics.return_value = [NS(tokens_per_s_per_gpu=900), NS(tokens_per_s_per_gpu=2700)]
    client.jobs.billing.return_value = NS(charged_credits=3)
    run("chat_sft/train.py", client, ["--yes"], monkeypatch)
    client.jobs.events.assert_called_once_with("job_1", follow=True)
    assert line in printed_while_training[0]


def test_chat_sft_names_the_models_layouts_when_it_has_not_the_one_asked(tmp_path, monkeypatch):
    for name in ("no_robots_train.jsonl", "no_robots_val.jsonl"):
        (tmp_path / name).write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    client = MagicMock()
    client.datasets.list.return_value = []
    client.models.return_value = [NS(id="qwen3.8-27b", layouts={"2x8-h100": NS(gpus=16), "4x8-h100": NS(gpus=32)})]
    with pytest.raises(SystemExit, match="trains on 2x8-h100, 4x8-h100"):
        run("chat_sft/train.py", client, ["--layout", "auto"], monkeypatch)


def test_math_rft_prints_a_first_round_that_has_no_round_time(tmp_path, monkeypatch, capsys):
    data = tmp_path / "prompts.jsonl"
    data.write_text('{"prompt": "1 + 1", "answer": "2"}\n')
    client = MagicMock()
    client.datasets.list.return_value = []
    client.datasets.wait.return_value = NS(id="ds_1", samples=1)
    client.jobs.estimate.return_value = NS(credits=1)
    client.jobs.create.return_value = NS(id="job_1")
    client.jobs.wait.return_value = NS(state="succeeded", error=None)
    client.jobs.metrics.return_value = [
        NS(step=1, reward_mean=0.746, zero_variance_fraction=0.1, completion_length_mean=300, generation_s=24.6, round_time_s=None),
        NS(step=2, reward_mean=0.8, zero_variance_fraction=0.1, completion_length_mean=280, generation_s=20.0, round_time_s=41.0)]
    client.jobs.billing.return_value = NS(charged_credits=120)
    run("math_rft/train.py", client, ["--data", str(data), "--yes"], monkeypatch)
    out = capsys.readouterr().out
    assert "sampling 25 s\n" in out and "sampling 20 s of 41 s\n" in out and "$1.20 charged" in out


@pytest.mark.parametrize("recipe, argv", [("math_rft/train.py", ["--data", "prompts.jsonl", "--yes"]), ("sft_basic.py", ["--yes"])])
def test_a_failed_job_ends_the_basic_recipes_with_its_error_too(recipe, argv, tmp_path, monkeypatch):
    (tmp_path / "prompts.jsonl").write_text('{"prompt": "1 + 1", "answer": "2"}\n')
    monkeypatch.chdir(tmp_path)
    client = MagicMock()
    client.datasets.list.return_value = [NS(id="ds_t", name="OpenThoughts Math (subset)", sample=True, state="ready"),
                                         NS(id="ds_v", name="OpenThoughts Math (validation subset)", sample=True, state="ready")]
    client.datasets.wait.return_value = NS(id="ds_1", samples=1)
    client.jobs.estimate.return_value = NS(steps=20, tokens=1000, credits=1)
    client.jobs.create.return_value = NS(id="job_1")
    client.jobs.wait.return_value = NS(state="failed", error="GPU initialization failed")
    with pytest.raises(SystemExit, match="job_1 failed: GPU initialization failed"):
        run(recipe, client, argv, monkeypatch)


def test_math_rl_starts_its_problems_again_when_the_steps_need_more(monkeypatch):
    import datasets

    problems = datasets.Dataset.from_list([{"problem": "1 + 1", "solution": "\\boxed{2}"}])
    monkeypatch.setattr("tinker_cookbook.recipes.math_rl.math_env._get_hendrycks_math_train", lambda: problems)
    monkeypatch.setattr("tinker_cookbook.recipes.math_rl.math_env._get_hendrycks_math_test", lambda: [])
    live = MagicMock()
    live.sample.return_value = [(NS(text="\\boxed{2}", finish_reason="stop"),)]
    live.train.return_value = STEP
    client = MagicMock()
    client.rl.connect.return_value.__enter__.return_value = live
    run("math_rl/train.py", client, ["rl_1", "--steps", "2", "--problems", "1"], monkeypatch)
    assert live.train.call_count == 2


def test_code_rl_starts_its_problems_again_when_the_steps_need_more(monkeypatch):
    task = NS(problem="Print the input.", tests=[])
    monkeypatch.setattr("tinker_cookbook.recipes.code_rl.code_env.load_deepcoder_tasks", lambda split: [task])

    async def passes(tests, code):
        return True, {"run_result": {"stdout": ""}}

    monkeypatch.setattr("tinker_cookbook.recipes.code_rl.code_grading.sandbox_check_correctness", passes)
    live = MagicMock()
    live.sample.return_value = [(NS(text="```python\nprint(input())\n```"),)]
    live.train.return_value = STEP
    client = MagicMock()
    client.rl.connect.return_value.__enter__.return_value = live
    run("code_rl/train.py", client, ["rl_1", "--steps", "2", "--problems", "1", "--episodes", "1"], monkeypatch)
    assert [len(call.kwargs["episodes"]) for call in live.train.call_args_list] == [1, 1]


def tinker_loop(monkeypatch, model):
    """Run recipes/tinker_loop.py against a stand-in for the tinker SDK whose RL run trains `model`."""
    tinker = MagicMock()
    tinker.RestClient.return_value.get_server_capabilities.return_value.result.return_value.supported_models = [NS(model_name=model)]
    service = tinker.ServiceClient.return_value
    service.__enter__.return_value = service
    sampler = service.create_lora_training_client.return_value.result.return_value.save_weights_and_get_sampling_client.return_value.result.return_value
    sampler.sample.return_value.result.return_value.sequences = [NS(tokens=[5], logprobs=[-0.1])]
    monkeypatch.setitem(sys.modules, "tinker", tinker)
    monkeypatch.setattr("tinker_cookbook.renderers.get_renderer", MagicMock())
    runpy.run_path(str(RECIPES / "tinker_loop.py"), run_name="__main__")
    return service


def test_tinker_loop_ends_its_session_so_the_run_stops_at_once(monkeypatch):
    service = tinker_loop(monkeypatch, "Qwen/Qwen3.8-27B")
    service.__exit__.assert_called_once()


def test_tinker_loop_refuses_an_rl_run_of_another_model(monkeypatch):
    with pytest.raises(SystemExit, match="trains google/gemma-4-26B-A4B-it"):
        tinker_loop(monkeypatch, "google/gemma-4-26B-A4B-it")


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
