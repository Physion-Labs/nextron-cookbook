import runpy
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

RECIPES = Path(__file__).parents[1] / "recipes"


def run(recipe, client, argv, monkeypatch):
    monkeypatch.setitem(sys.modules, "nextron_cloud", NS(NexTron=lambda: client))
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
