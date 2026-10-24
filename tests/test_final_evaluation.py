"""Freeze inputs and recover only intact per-model results before final test."""

import copy
import csv
import json

import numpy as np
import pytest

from recsys.final_evaluation import check_source, completed_result, finish, inventory, read_selection, resolve_sources, save_arrays, verify_snapshot_metrics, verify_state
from recsys.metrics import per_user_ranking
from recsys.io import sha256, write_json


def frozen_choice():
    return {"test_driven_reselection_allowed": False, "primary_family": "primary", "seeds": [42],
        "models": [{"name": "sft", "family": "sft", "source_relative_to_drive_base": "scaling/best.pt"},
            {"name": "rl42", "family": "primary", "seed": 42,
             "source_relative_to_drive_base": "branches/branch42/best.pt"}]}


@pytest.mark.parametrize("change", ["reselection", "duplicate", "missing_seed", "escape"])
def test_reject_inconsistent_selection(tmp_path, change):
    choice = frozen_choice()
    if change == "reselection":
        choice["test_driven_reselection_allowed"] = True
    elif change == "duplicate":
        choice["models"].append(choice["models"][1])
    elif change == "missing_seed":
        choice["seeds"].append(123)
    else:
        choice["models"][1]["source_relative_to_drive_base"] = "../other/best.pt"
    path = tmp_path / "selection.json"
    write_json(path, choice)
    with pytest.raises(ValueError):
        read_selection(path)


def test_inventory_reports_all_missing_and_empty_inputs_without_loading_test(tmp_path):
    (tmp_path / "scaling").mkdir()
    (tmp_path / "scaling/best.pt").touch()
    with pytest.raises(FileNotFoundError) as error:
        inventory(tmp_path, frozen_choice())
    assert "scaling/best.pt" in str(error.value)
    assert "branches/branch42/best.pt" in str(error.value)
    assert "prepared/manifest.json" in str(error.value)


def make_available_inputs(base, choice):
    for entry in choice["models"]:
        path = base / entry["source_relative_to_drive_base"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
    rm = base / "artifacts/rm/best.pt"
    rm.parent.mkdir(parents=True)
    rm.write_bytes(b"reward")
    for label in ["prepared", "scaling_v1/data/large"]:
        write_json(base / label / "manifest.json", {"file_sha256": {}})


def test_only_diagnostics_can_be_missing_and_original_selection_does_not_change(tmp_path):
    choice = frozen_choice()
    choice["models"].append({"name": "ce123", "family": "ce03_fixed", "seed": 123,
        "expected_step": 2500, "source_relative_to_drive_base": "ablation/branches/ce123/last.pt"})
    original = copy.deepcopy(choice)
    make_available_inputs(tmp_path, choice)
    missing = tmp_path / choice["models"][-1]["source_relative_to_drive_base"]
    missing.unlink()
    with pytest.raises(FileNotFoundError):
        inventory(tmp_path, choice)
    inventory(tmp_path, choice, allow_missing_diagnostics=True)
    assert choice == original
    (tmp_path / choice["models"][1]["source_relative_to_drive_base"]).unlink()
    with pytest.raises(FileNotFoundError, match="branch42"):
        inventory(tmp_path, choice, allow_missing_diagnostics=True)


def test_recovery_uses_known_local_layout_or_exact_frozen_compact_hash(tmp_path):
    choice = frozen_choice()
    entry = choice["models"][1]
    entry.update(source_relative_to_drive_base="study/branches/branch42/last.pt", expected_step=2500,
                 validation_evaluation_alias="branch42_floor_0990", validation_evaluated_file_sha256="not-the-hash")
    base, local = tmp_path / "drive", tmp_path / "colab"
    compact = base / "study/branches/branch42/best_floor_0990.pt"
    compact.parent.mkdir(parents=True)
    compact.write_bytes(b"wrong snapshot")
    sources, missing = resolve_sources(base, choice, [local])
    assert "rl42" not in sources
    entry["validation_evaluated_file_sha256"] = sha256(compact)
    sources, _ = resolve_sources(base, choice, [local])
    assert sources["rl42"]["path"] == compact
    assert sources["rl42"]["recovery"] == "hash_locked_validation_copy"
    compact.unlink()
    last = local / "artifacts_study/branches/branch42/last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"original full checkpoint")
    sources, _ = resolve_sources(base, choice, [local])
    assert sources["rl42"]["path"] == last
    assert sources["rl42"]["recovery"] == "same_checkpoint_local_copy"


def test_best_recovery_rejects_different_weights_even_when_path_matches(tmp_path):
    choice = frozen_choice()
    entry = choice["models"][1]
    entry.update(source_relative_to_drive_base="study/branches/branch42/best.pt", source_file_sha256="frozen-hash")
    path = tmp_path / "colab/artifacts_study/branches/branch42/best.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"different weights")
    sources, _ = resolve_sources(tmp_path / "drive", choice, [tmp_path / "colab"])
    assert "rl42" not in sources


def test_snapshot_recovery_never_substitutes_a_different_update(tmp_path):
    choice = frozen_choice()
    choice["models"][1].update(source_relative_to_drive_base="study/branches/branch42/last.pt", expected_step=2500)
    folder = tmp_path / "study/branches/branch42"
    folder.mkdir(parents=True)
    (folder / "step_0002499.pt").write_bytes(b"other step")
    sources, _ = resolve_sources(tmp_path, choice)
    assert "rl42" not in sources
    (folder / "step_0002500.pt").write_bytes(b"frozen step")
    sources, _ = resolve_sources(tmp_path, choice)
    assert sources["rl42"]["recovery"] == "same_update_snapshot"


def test_snapshot_requires_matching_saved_validation_metrics():
    entry = {"name": "control", "validation_ndcg@10": .112, "validation_rm_reward@10": .56}
    state = {"selection": {"metrics": {"ndcg@10": .112, "rm_reward@10": .56000001}}}
    verify_snapshot_metrics(entry, state)
    with pytest.raises(ValueError, match="lacks frozen validation"):
        verify_snapshot_metrics(entry, {})
    state["selection"]["metrics"]["ndcg@10"] = .113
    with pytest.raises(ValueError, match="validation metrics differ"):
        verify_snapshot_metrics(entry, state)


@pytest.mark.parametrize("primary", [False, True])
def test_wrong_step_is_recorded_only_for_optional_diagnostics(tmp_path, primary):
    path = tmp_path / "last.pt"
    path.write_bytes(b"original checkpoint")
    entry = {"name": "ce10_fixed_s42", "family": "primary" if primary else "ce10_fixed",
             "seed": 42, "expected_step": 2500}
    selection = {"primary_family": "primary", "source_identity": {}}
    found = {"path": path, "expected_sha256": None, "recovery": "original"}
    def loader(_):
        return {"run_spec": {}, "model_spec": {"kind": "policy"}, "step": 1900}
    if primary:
        with pytest.raises(ValueError, match="found policy at step 1900"):
            check_source(entry, found, selection, loader, allow_missing_diagnostics=True)
    else:
        state, digest, unavailable = check_source(entry, found, selection, loader, allow_missing_diagnostics=True)
        assert state is None and digest is None
        assert unavailable["actual_step"] == 1900 and unavailable["expected_step"] == 2500
        assert unavailable["reason"] == "checkpoint_failed_verification_before_test"
        with pytest.raises(ValueError):
            check_source(entry, found, selection, loader, allow_missing_diagnostics=False)


def test_incomplete_family_is_listed_without_surviving_seed_average(tmp_path):
    root, backup = tmp_path / "local", tmp_path / "drive"
    output = root / "evaluation/test"
    output.mkdir(parents=True)
    data = {"users": np.array([1, 2]), "targets": np.array([3, 4]), "rewards": np.array([.4, .6])}
    slate = np.array([[3, 4], [4, 3]])
    values = {**per_user_ranking(slate, data["targets"]), "rm_reward@10": np.array([.5, .5])}
    metrics = {}
    for name in ["sft", "rl42", "ce42"]:
        save_arrays(output / f"{name}_per_user.npz", **data, slate=slate, **values,
                    observed_prediction=np.array([.4, .6]))
        metrics[name] = {"ranking": {k: float(v.mean()) for k, v in values.items() if k != "rm_reward@10"},
                         "rm_reward@10": .5, "behavior@10": {"coverage": .2, "seen_fraction": .1}}
    choice = frozen_choice()
    choice["source_identity"] = {"manifest_sha256": "data", "sft_sha256": "sft", "rm_sha256": "rm"}
    for seed in [42, 123, 2026]:
        choice["models"].append({"name": f"ce{seed}", "seed": seed, "family": "ce03_fixed"})
    missing = [{"name": "ce123"}, {"name": "ce2026"}]
    checked = {"identity": {"selection_sha256": "frozen"}, "models": {k: {} for k in metrics},
               "missing_models": missing, "reward_transfer": None}
    write_json(root / "data/expanded/manifest.json", {"global_train_reward": .5})
    np.save(root / "data/expanded/item_rewards.npy", np.full(5, .5))
    finish(root, backup, choice, checked, data, metrics)
    with (backup / "reports/seed_summary.csv").open() as stream:
        control = next(row for row in csv.DictReader(stream) if row["family"] == "ce03_fixed")
    assert control["n_seeds"] == "1" and control["expected_n_seeds"] == "3"
    assert control["missing_seeds"] == "123/2026" and control["complete_family"] == "False"
    assert control["ndcg@10_mean"] == "" and control["rm_reward@10_change_pct_mean"] == ""
    result = json.loads((backup / "evaluation/test/summary.json").read_text())
    assert result["missing_models"] == missing and result["all_requested_models_evaluated"] is False


@pytest.mark.parametrize("change", ["step", "seed", "code", "dependencies", "branch", "stage", "manifest"])
def test_fixed_checkpoint_requires_frozen_metadata_even_without_a_compact_file_hash(change):
    entry = {"name": "rl42", "family": "rl_ce03_fixed", "seed": 42, "expected_step": 2500,
             "training_code_sha256": "original-code", "source_relative_to_drive_base": "branches/branch42/last.pt"}
    identity = {"manifest_sha256": "data", "sft_sha256": "sft", "rm_sha256": "rm"}
    state = {"model_spec": {"kind": "policy"}, "step": 2500,
        "run_spec": {"manifest_sha256": "data", "code_sha256": "original-code", "seed": 42,
            "dependencies": {"init_sha256": "sft", "reward_sha256": "rm"}, "stage": "rl",
            "settings": {"name": "branch42"}}}
    verify_state(entry, state, identity)
    changed = copy.deepcopy(state)
    if change == "step":
        changed["step"] = 2499
    else:
        key = {"code": "code_sha256", "manifest": "manifest_sha256"}.get(change, change)
        if change == "branch":
            changed["run_spec"]["settings"]["name"] = "other"
        else:
            changed["run_spec"][key] = "other"
    with pytest.raises(ValueError):
        verify_state(entry, changed, identity)


def test_resume_recovers_verified_arrays_and_rejects_corruption(tmp_path):
    local, backup = tmp_path / "local", tmp_path / "backup"
    folder = backup / "evaluation/test"
    folder.mkdir(parents=True)
    write_json(folder / "sft.json", {"rm_reward@10": .5})
    save_arrays(folder / "sft_per_user.npz", users=np.array([1, 2]), targets=np.array([3, 4]))
    files = {p.name: sha256(p) for p in folder.iterdir()}
    write_json(folder / "sft.done.json", {"files": files})
    assert completed_result(local, backup, "sft") == {"rm_reward@10": .5}
    with np.load(local / "evaluation/test/sft_per_user.npz") as z:
        assert z["users"].tolist() == [1, 2]
    (local / "evaluation/test/sft_per_user.npz").write_bytes(b"broken")
    with pytest.raises(ValueError, match="changed saved result"):
        completed_result(local, backup, "sft")
    assert completed_result(local, backup, "not_completed") is None
