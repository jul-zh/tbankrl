"""Small CPU tests run in Colab before the actual scaling study."""

import json
import signal

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from recsys.budget import new_deadline
from recsys.scaling import prepare_scaling
from recsys.sft_study import fit, lr_multiplier, run
from recsys.training import read_checkpoint
from test_scaling import scaling_fixture


def tiny_plan():
    return {"seed": 42, "overall_minutes": 2, "evaluation_reserve_minutes": .01, "checkpoint_minutes": 0,
            "pilot_steps": 2, "training": {"batch_size": 4, "learning_rate": .001, "min_lr_ratio": .1,
            "warmup_passes": 0, "max_passes": .25, "eval_passes": .1, "patience": 99,
            "min_delta": .0001, "max_minutes": 1, "eval_batch_size": 4, "train_probe_examples": 4},
            "branches": [{"name": "small_128", "dataset": "small", "model": {"hidden": 16, "layers": 1, "heads": 2, "dropout": .1}},
                         {"name": "large_128", "dataset": "large", "model": {"hidden": 16, "layers": 1, "heads": 2, "dropout": .1}}]}


def test_scheduler_and_scaling_end_to_end(tmp_path, monkeypatch):
    import recsys.sft_study as study
    torch.set_num_threads(1)
    monkeypatch.setattr(study, "device", lambda: torch.device("cpu"))
    csv_path, source = scaling_fixture(tmp_path)
    data = tmp_path / "data"
    prepare_scaling(csv_path, source, data, {"max_users": 7, "max_added_examples": 1000})
    plan = tiny_plan()
    root, backup = tmp_path / "runs", tmp_path / "backup"
    run(data, root, backup, plan, allow_cpu=True)
    journal = json.loads((root / "study.json").read_text())
    assert journal["completed"] == ["small_128", "large_128"]
    result = json.loads((root / "evaluation/summary.json").read_text())
    assert "large_128_minus_small_128" in result["comparisons"]
    assert not (root / "evaluation/test").exists()
    assert (backup / "large_128/last.pt").exists()
    before = (root / "small_128/last.pt").stat().st_mtime_ns
    run(data, root, backup, plan, allow_cpu=True)
    assert (root / "small_128/last.pt").stat().st_mtime_ns == before
    assert json.loads((root / "study.json").read_text())["deadline"] == journal["deadline"]
    assert lr_multiplier(0, 10, 100, .1) == .1
    assert lr_multiplier(10, 10, 100, .1) == 1
    assert lr_multiplier(100, 10, 100, .1) == pytest.approx(.1)


@pytest.mark.parametrize("interruption", ["checkpoint", "signal"])
def test_study_resume_restores_optimizer_scheduler_and_rng(tmp_path, monkeypatch, interruption):
    import recsys.sft_study as study
    torch.set_num_threads(1)
    monkeypatch.setattr(study, "device", lambda: torch.device("cpu"))
    _, source = scaling_fixture(tmp_path)
    plan = tiny_plan()
    branch = plan["branches"][0]
    deadline = new_deadline(2)
    fit(source, tmp_path / "full", tmp_path / "full_backup", branch, plan, deadline)
    original = study.atomic_checkpoint
    original_update = study.update
    interrupted = False

    def interrupt(path, state):
        nonlocal interrupted
        original(path, state)
        if path.name == "last.pt" and state["step"] == 1 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("simulated interruption at an update boundary")

    def signal_after_update(*args, **kwargs):
        nonlocal interrupted
        success = original_update(*args, **kwargs)
        if success and not interrupted:
            interrupted = True
            signal.raise_signal(signal.SIGINT)
        return success

    if interruption == "checkpoint":
        monkeypatch.setattr(study, "atomic_checkpoint", interrupt)
    else:
        monkeypatch.setattr(study, "update", signal_after_update)
    with pytest.raises(KeyboardInterrupt):
        fit(source, tmp_path / "resumed", tmp_path / "resume_backup", branch, plan, deadline)
    monkeypatch.setattr(study, "atomic_checkpoint", original)
    monkeypatch.setattr(study, "update", original_update)
    fit(source, tmp_path / "resumed", tmp_path / "resume_backup", branch, plan, deadline)
    a = read_checkpoint(tmp_path / "full/last.pt", torch.device("cpu"))
    b = read_checkpoint(tmp_path / "resumed/last.pt", torch.device("cpu"))
    assert a["step"] == b["step"] and a["stop_reason"] == b["stop_reason"] == "max_steps"
    assert a["scheduler"] == b["scheduler"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
