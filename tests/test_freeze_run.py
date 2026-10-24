"""Фиксация нового обучения не подменяет шаги и опубликованный план."""

import copy
import runpy
from pathlib import Path

import pytest

from recsys.io import sha256, write_json


freeze = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/freeze_run.py"))["freeze"]


def fixture(base, last_step):
    for folder in [base / "prepared", base / "scaling_v1/data/large"]:
        write_json(folder / "manifest.json", {"format_version": 1, "file_sha256": {}})
    sft = base / "scaling_v1/large_128/best.pt"
    rm = base / "artifacts/rm/best.pt"
    primary = base / "rl_preserve_v1/branches/anchor03_s42/best.pt"
    last = primary.with_name("last.pt")
    for path in [sft, rm, primary, last]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str(path).encode())
    data_hash = sha256(base / "scaling_v1/data/large/manifest.json")
    template = {"models": [
        {"name": "sft", "family": "sft", "source_relative_to_drive_base": str(sft.relative_to(base)), "expected_step": 45612},
        {"name": "rl_s42", "family": "rl_ce03_selected", "seed": 42,
         "source_relative_to_drive_base": str(primary.relative_to(base)), "expected_step": 2300, "validation_ndcg@10": .12},
        {"name": "fixed_s42", "family": "rl_ce03_fixed", "seed": 42,
         "source_relative_to_drive_base": str(last.relative_to(base)), "expected_step": 2500}]}
    def loader(path):
        return {"model_spec": {"kind": "policy"}, "step": last_step if path == last else 13,
                "run_spec": {"manifest_sha256": data_hash, "seed": 42, "code_sha256": "new-training-code",
                             "dependencies": {"init_sha256": sha256(sft), "reward_sha256": sha256(rm)}}}
    return template, loader


def test_new_validation_choice_does_not_reuse_old_metrics_or_modify_template(tmp_path):
    template, loader = fixture(tmp_path, 2500)
    original = copy.deepcopy(template)
    result = freeze(tmp_path, template, loader)
    assert template == original
    assert result["models"][1]["expected_step"] == 13
    assert "validation_ndcg@10" not in result["models"][1]
    assert result["models"][1]["training_code_sha256"] == "new-training-code"
    assert result["models"][2]["expected_step"] == 2500


def test_incomplete_last_update_is_not_substituted(tmp_path):
    template, loader = fixture(tmp_path, 1900)
    with pytest.raises(ValueError, match="требуется 2500"):
        freeze(tmp_path, template, loader)
