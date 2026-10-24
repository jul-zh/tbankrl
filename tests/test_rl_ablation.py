"""Matched comparisons must keep direction, identities and training settings."""

import copy
import io
import json
from zipfile import ZipFile

import numpy as np
import pytest

from recsys.io import sha256, write_json
from recsys.rl_ablation import compare_exports, validate_reference


def ablation_fixture(tmp_path):
    identity = {k: k + "_hash" for k in ["sft_sha256", "rm_sha256", "manifest_sha256", "reward_source_manifest_sha256"]}
    training = {"max_steps": 2500, "max_minutes": 12, "beta": .3, "ce_weight": .3}
    common = {"seed": 42, "relevance_fraction": 1., "selection_fractions": [1.]}
    original = {**common, "training": training, "branches": [{"name": "anchor03_s42", "seed": 42, "family": "anchor03"}]}
    plan = {**common, "training": {**training, "reinforce_weight": 0.},
            "branches": [{"name": "ce03_s42", "seed": 42, "family": "ce03", "reference_branch": "anchor03_s42"}]}
    users = np.array([1, 2, 3])
    base = {"users": users, "targets": users + 3, "slate": (users + 3)[:, None],
            "ndcg@10": np.full(3, .1), "rm_reward@10": np.full(3, .5)}
    rl = {**base, "ndcg@10": np.full(3, .102), "rm_reward@10": np.full(3, .55)}
    control = {**base, "ndcg@10": np.full(3, .101), "rm_reward@10": np.full(3, .52)}

    def summary(name, values):
        return {"split": "validation", "n_users": 3, "manifest_sha256": identity["manifest_sha256"],
            "reference_checkpoint_sha256": identity["sft_sha256"], "reward_checkpoint_sha256": identity["rm_sha256"],
            "models": {"sft": {"ranking": {"ndcg@10": .1}, "rm_reward@10": .5}, name: {"checkpoint_step": 2500}},
            "selection_by_constraint": {name: {"floor_1000": {"model": name}}}, "last_checkpoint_aliases": {name: name}}

    archive = tmp_path / "previous.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("runs/experiment.json", json.dumps({"identity": {**identity, "plan": original, "code_sha256": "before"}}))
        z.writestr("runs/evaluation/validation/summary.json", json.dumps(summary("anchor03_s42", rl)))
        for name, values in [("sft", base), ("anchor03_s42", rl)]:
            output = io.BytesIO()
            np.savez(output, **values)
            z.writestr(f"runs/evaluation/validation/{name}_per_user.npz", output.getvalue())
    plan["comparison_archive_sha256"] = sha256(archive)
    root = tmp_path / "runs"
    output = root / "evaluation/validation"
    output.mkdir(parents=True)
    (root / "reports").mkdir()
    write_json(root / "experiment.json", {"identity": {**identity, "plan": plan, "code_sha256": "after"}})
    write_json(output / "summary.json", summary("ce03_s42", control))
    for name, values in [("sft", base), ("ce03_s42", control)]:
        np.savez(output / (name + "_per_user.npz"), **values)
    return root, archive, plan, identity


def test_ablation_pairs_same_users_and_uses_rl_minus_control(tmp_path):
    root, archive, plan, _ = ablation_fixture(tmp_path)
    result = compare_exports(root, archive, plan)
    assert len(result["comparisons"]) == 2
    assert {r["selection"] for r in result["comparisons"]} == {"floor_1000", "fixed_2500"}
    paired = result["comparisons"][0]["paired_rl_minus_control"]
    assert paired["ndcg@10"]["mean_delta"] == pytest.approx(.001)
    assert paired["rm_reward@10"]["mean_delta"] == pytest.approx(.03)
    assert result["missing_or_incomplete"] == []
    assert (root / "reports/ablation.md").exists()
    path = root / "evaluation/validation/ce03_s42_per_user.npz"
    with np.load(path) as z:
        values = {k: z[k] for k in z.files}
    values["users"] = values["users"][::-1]
    np.savez(path, **values)
    with pytest.raises(AssertionError):
        compare_exports(root, archive, plan)


def test_ablation_rejects_changed_reference_or_training(tmp_path):
    _, archive, plan, identity = ablation_fixture(tmp_path)
    validate_reference(archive, plan, identity)
    wrong = copy.deepcopy(plan)
    wrong["training"]["ce_weight"] = 1.
    with pytest.raises(ValueError, match="Training settings"):
        validate_reference(archive, wrong, identity)
    wrong["comparison_archive_sha256"] = "wrong"
    with pytest.raises(ValueError, match="archive hash"):
        validate_reference(archive, wrong, identity)


def test_estimator_comparison_uses_candidate_minus_historical(tmp_path):
    root, archive, plan, identity = ablation_fixture(tmp_path)
    plan["comparison_kind"] = "estimator_study"
    plan["training"].update(reinforce_weight=1., objective_mode="full_catalog", beta=1., max_minutes=8)
    validate_reference(archive, plan, identity)
    result = compare_exports(root, archive, plan)
    paired = result["comparisons"][0]["paired_candidate_minus_historical"]
    assert paired["ndcg@10"]["mean_delta"] == pytest.approx(-.001)
    assert paired["rm_reward@10"]["mean_delta"] == pytest.approx(-.03)
    assert paired["rm_reward@10"]["ci95"][1] < 0
    assert result["seed_aggregates"][0]["candidate_reward_change_pct_mean"] == pytest.approx(4.)
    assert (root / "reports/historical.md").exists()
    wrong = copy.deepcopy(plan)
    wrong["training"]["ce_weight"] = 1.
    with pytest.raises(ValueError, match="Training settings"):
        validate_reference(archive, wrong, identity)
