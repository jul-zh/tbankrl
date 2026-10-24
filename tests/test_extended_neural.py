"""A tiny end-to-end extension; run in Colab before the long experiment."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from recsys.cli import main
from recsys.data import prepare
from recsys.extended import run
from recsys.io import sha256


def test_extended_preserves_originals_budget_and_checkpoint_protocol(tmp_path, monkeypatch):
    import recsys.training as training
    import recsys.extended as extended
    torch.set_num_threads(1)
    monkeypatch.setattr(training, "device", lambda: torch.device("cpu"))
    monkeypatch.setattr(extended, "device", lambda: torch.device("cpu"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    source_csv = tmp_path / "interactions.csv"
    pd.DataFrame([
        (u, 10 + (day + u) % 5, f"2021-01-{day:02d}", 20 + 10 * ((day + u) % 5))
        for u in range(4) for day in range(1, 21)
    ], columns=["user_id", "item_id", "last_watch_dt", "watched_pct"]).to_csv(source_csv, index=False)
    config = {"seed": 42, "context": 5, "min_train_events": 3, "max_users": 4,
              "max_train_examples": 100, "max_eval_users": 4, "train_fraction": .5, "validation_fraction": .25,
              "model": {"hidden": 16, "layers": 1, "heads": 2, "dropout": .1},
              "checkpoint_minutes": 0, "eval_batch_size": 4, "validation_train_limit": 4}
    for stage in ["sft", "rm", "rl"]:
        config[stage] = {"batch_size": 4, "learning_rate": .001, "max_steps": 2,
                         "max_minutes": 2, "eval_every": 1, "gamma": .95}
    data, source = tmp_path / "prepared", tmp_path / "original"
    prepare(source_csv, data, config)
    for stage in ["sft", "rm"]:
        main(["train", "--data", str(data), "--runs", str(source), "--stage", stage])
    original_hashes = {name: sha256(source / name / "best.pt") for name in ["sft", "rm"]}
    plan = {"overall_minutes": 2, "evaluation_reserve_minutes": .1, "checkpoint_minutes": 0,
            "seed": 42, "relevance_fraction": .95,
            "sft": {"name": "sft", "max_steps": 2, "max_minutes": .2, "learning_rate": .0001,
                    "min_learning_rate": .00001, "batch_size": 4, "eval_every": 2},
            "rl": [{"name": "rl_long", "beta": .3, "horizon": 1}],
            "rl_defaults": {"max_steps": 2, "max_minutes": .2, "learning_rate": .0001,
                            "min_learning_rate": .00001, "batch_size": 4, "eval_every": 2,
                            "gamma": .95, "repeat_penalty": 0.}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    args = SimpleNamespace(data=data, source_runs=source, runs=tmp_path / "extended",
                           backup=tmp_path / "backup", config=plan_path, allow_cpu=True)
    args.runs.mkdir()
    # Operational caps can shorten a run without editing the frozen plan or data.
    (args.runs / "runtime_limits.json").write_text(json.dumps({"sft": 1, "rl_long": 1}))
    run(args)
    journal = json.loads((args.runs / "experiment.json").read_text())
    assert journal["completed"] == ["sft", "rl_long"]
    assert journal["deadline"] - journal["started_at"] == pytest.approx(120, abs=.1)
    for name, expected in original_hashes.items():
        assert sha256(source / name / "best.pt") == expected
    for name in ["sft", "rl_long"]:
        assert (args.backup / name / "last.pt").exists()
        assert (args.backup / name / "step_0000000.pt").exists()
        assert (args.backup / name / "step_0000001.pt").exists()
        assert not (args.runs / name / "step_0000002.pt").exists()
        timing = json.loads((args.runs / name / "timing.json").read_text())
        assert timing["effective_max_steps"] == timing["completed_steps"] == 1
        stored = training.read_checkpoint(args.runs / name / "last.pt", torch.device("cpu"))
        assert stored["run_spec"]["settings"]["max_steps"] == 2
    assert (args.backup / "runtime_limits.json").exists()
    summary = json.loads((args.runs / "evaluation/validation/summary.json").read_text())
    assert {"sft", "sft_last", "rl_long", "rl_long_last", "rerank"} <= summary["models"].keys()
    reference = summary["models"]["sft"]["ranking"]["ndcg@10"]
    selected = summary["models"]["rl_long"]["ranking"]["ndcg@10"]
    assert selected >= .95 * reference
    assert not (args.runs / "evaluation/test").exists()
    # Re-running a finished experiment does not restart its training budget.
    previous = (args.runs / "sft/last.pt").stat().st_mtime_ns
    run(args)
    again = json.loads((args.runs / "experiment.json").read_text())
    assert again["deadline"] == journal["deadline"]
    assert (args.runs / "sft/last.pt").stat().st_mtime_ns == previous
