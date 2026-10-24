"""Tiny inference-only end-to-end check; no optimizer or training allowed."""

import copy
import csv
import json
import shutil
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from recsys.final_evaluation import export, preflight, run
from recsys.io import sha256, write_json
from recsys.models import Policy, RewardModel
from recsys.scaling import prepare_scaling
from recsys.training import atomic_checkpoint
from test_scaling import scaling_fixture


@pytest.mark.parametrize("missing_diagnostic", [False, True, "wrong_step"])
def test_final_evaluation_never_trains_and_resumes_completed_models(tmp_path, monkeypatch, missing_diagnostic):
    torch.set_num_threads(1)
    csv_path, original = scaling_fixture(tmp_path)
    base = tmp_path / "drive"
    shutil.copytree(original, base / "prepared")
    scaled = tmp_path / "scaled"
    prepare_scaling(csv_path, original, scaled, {"max_users": 7, "max_added_examples": 1000})
    shutil.copytree(scaled / "large", base / "scaling_v1/data/large")
    manifest = json.loads((scaled / "large/manifest.json").read_text())
    spec = {"kind": "policy", "kwargs": {"vocab_size": manifest["vocab_size"], "context": 6,
        "hidden": 16, "layers": 1, "heads": 2, "dropout": .1}}
    policy = Policy(**spec["kwargs"])
    reward_spec = {"kind": "reward", "kwargs": {"vocab_size": manifest["vocab_size"], "hidden": 8}}
    reward = RewardModel(**reward_spec["kwargs"])
    sft_path, rm_path = base / "scaling_v1/sft/best.pt", base / "artifacts/rm/best.pt"
    for path in [sft_path, rm_path]:
        path.parent.mkdir(parents=True)
    atomic_checkpoint(sft_path, {"model": policy.state_dict(), "model_spec": spec, "step": 3,
        "run_spec": {"stage": "sft", "manifest_sha256": sha256(scaled / "large/manifest.json")}})
    atomic_checkpoint(rm_path, {"model": reward.state_dict(), "model_spec": reward_spec, "step": 1,
        "run_spec": {"stage": "rm", "manifest_sha256": sha256(original / "manifest.json")}})
    identity = {"manifest_sha256": sha256(scaled / "large/manifest.json"), "sft_sha256": sha256(sft_path),
        "rm_sha256": sha256(rm_path), "reward_source_manifest_sha256": sha256(original / "manifest.json")}
    choices = [{"name": "sft", "family": "sft", "source_relative_to_drive_base": "scaling_v1/sft/best.pt",
                "source_file_sha256": sha256(sft_path), "expected_step": 3}]
    for seed in [42, 123, 2026]:
        name = f"rl{seed}"
        path = base / "branches" / name / "best.pt"
        path.parent.mkdir(parents=True)
        atomic_checkpoint(path, {"model": copy.deepcopy(policy.state_dict()), "model_spec": spec, "step": 2,
            "run_spec": {"stage": "rl", "seed": seed, "code_sha256": "historical", "settings": {"name": name},
                "manifest_sha256": identity["manifest_sha256"],
                "dependencies": {"init_sha256": identity["sft_sha256"], "reward_sha256": identity["rm_sha256"]}}})
        choices.append({"name": name, "family": "rl_ce03_selected", "seed": seed,
            "source_relative_to_drive_base": str(path.relative_to(base)), "expected_step": 2,
            "source_file_sha256": sha256(path), "training_code_sha256": "historical"})
    for seed in [42, 123, 2026]:
        name = f"ce{seed}"
        path = base / "branches" / name / "last.pt"
        path.parent.mkdir(parents=True)
        atomic_checkpoint(path, {"model": copy.deepcopy(policy.state_dict()), "model_spec": spec, "step": 2,
            "run_spec": {"stage": "sft_control", "seed": seed, "code_sha256": "historical",
                "settings": {"name": name}, "manifest_sha256": identity["manifest_sha256"],
                "dependencies": {"init_sha256": identity["sft_sha256"], "reward_sha256": identity["rm_sha256"]}}})
        choices.append({"name": name, "family": "ce03_fixed", "seed": seed,
            "source_relative_to_drive_base": str(path.relative_to(base)), "expected_step": 2,
            "training_code_sha256": "historical"})
        if missing_diagnostic and seed != 42:
            if missing_diagnostic == "wrong_step":
                state = torch.load(path, map_location="cpu", weights_only=False)
                state["step"] = 1
                atomic_checkpoint(path, state)
            else:
                path.unlink()
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, {"models": choices, "source_identity": identity, "seeds": [42, 123, 2026],
        "primary_family": "rl_ce03_selected", "rerank_lambdas": [.25], "test_driven_reselection_allowed": False})
    before = {p: sha256(p) for p in base.rglob("*.pt")}
    def no_optimizer(*args, **kwargs):
        raise AssertionError("Final evaluation must never construct an optimizer")
    monkeypatch.setattr(torch.optim, "AdamW", no_optimizer)
    args = SimpleNamespace(base=base, runs=tmp_path / "runs", backup=base / "final_test_v1",
        selection=selection_path, allow_cpu=True, allow_missing_diagnostics=bool(missing_diagnostic),
        max_minutes=2., model_minutes=1.)
    preflight(args)
    run(args)
    summary = json.loads((args.backup / "evaluation/test/summary.json").read_text())
    assert summary["split"] == "test" and len(summary["models"]) == (7 if missing_diagnostic else 9)
    assert summary["test_driven_reselection_allowed"] is False
    assert summary["test_previously_examined"] is True
    progress = json.loads((args.backup / "progress.json").read_text())
    assert progress["complete"] is (not bool(missing_diagnostic))
    assert progress["available_models_complete"] and progress["primary_complete"]
    assert json.loads((args.backup / "FINAL_SELECTION.json").read_text()) == json.loads(selection_path.read_text())
    if missing_diagnostic:
        assert {m["name"] for m in summary["missing_models"]} == {"ce123", "ce2026"}
        if missing_diagnostic == "wrong_step":
            assert all(m["actual_step"] == 1 and m["expected_step"] == 2 for m in summary["missing_models"])
        with (args.backup / "reports/seed_summary.csv").open() as stream:
            control = next(row for row in csv.DictReader(stream) if row["family"] == "ce03_fixed")
        assert control["n_seeds"] == "1" and control["expected_n_seeds"] == "3"
        assert control["ndcg@10_mean"] == "" and control["complete_family"] == "False"
    for path, expected in before.items():
        assert sha256(path) == expected
    def no_scoring(*args, **kwargs):
        raise AssertionError("A completed model must not be scored again")
    monkeypatch.setattr("recsys.evaluation.score_policy", no_scoring)
    run(args)
    export(args)
    assert (args.backup / "results_final_test_v1.zip").stat().st_size > 0


def test_policy_evaluation_obeys_deadline_before_forward():
    import numpy as np
    from recsys.evaluation import score_policy
    data = {"targets": np.array([3])}
    with pytest.raises(TimeoutError, match="deadline"):
        score_policy(None, None, None, data, None, torch.device("cpu"), 1, deadline=time.monotonic() - 1)
