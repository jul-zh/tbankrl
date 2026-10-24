import json
from types import SimpleNamespace

import pytest

from recsys.reporting import report


def model(ndcg, reward, checkpoint):
    return {"ranking": {"ndcg@10": ndcg, "recall@10": .5}, "rm_reward@10": reward,
            "behavior@10": {"coverage": .3, "seen_fraction": .2}, "checkpoint_sha256": checkpoint,
            "stage": "sft" if checkpoint == "sft" else "rl"}


def summary():
    return {"manifest_sha256": "synthetic-protocol", "n_users": 10,
            "models": {"sft": model(.5, .5, "sft"), "rl_good": model(.49, .7, "good"),
                       "rl_bad": model(.3, .9, "bad")}, "reward_model_evaluation": {}}


def test_report_selects_reward_under_relevance_constraint(tmp_path):
    path = tmp_path / "runs/evaluation/validation"
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(summary()))
    args = SimpleNamespace(runs=tmp_path / "runs", output=tmp_path / "report")
    report(args)
    selection = json.loads((args.output / "selection.json").read_text())
    assert selection["selected_rl"] == "rl_good"
    assert (args.output / "tradeoff_and_rollout.png").is_file()


def test_report_does_not_call_identical_sft_a_reward_improvement(tmp_path):
    path = tmp_path / "runs/evaluation/validation"
    path.mkdir(parents=True)
    content = summary()
    content["models"]["rl_good"]["rm_reward@10"] = .5
    (path / "summary.json").write_text(json.dumps(content))
    args = SimpleNamespace(runs=tmp_path / "runs", output=tmp_path / "report")
    report(args)
    assert json.loads((args.output / "selection.json").read_text())["selected_rl"] is None


def test_report_can_require_zero_ndcg_loss(tmp_path):
    path = tmp_path / "runs/evaluation/validation"
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(summary()))
    args = SimpleNamespace(runs=tmp_path / "runs", output=tmp_path / "report", relevance_fraction=1.)
    report(args)
    result = json.loads((args.output / "selection.json").read_text())
    assert result["selected_rl"] is None
    assert result["validation_ndcg_floor"] == .5
    assert "100%" in result["criterion"]


@pytest.mark.parametrize("change", ["protocol", "checkpoint"])
def test_report_rejects_inconsistent_validation_test(tmp_path, change):
    base = tmp_path / "runs/evaluation"
    for split in ("validation", "test"):
        path = base / split
        path.mkdir(parents=True)
        content = summary()
        if split == "test":
            if change == "protocol":
                content["manifest_sha256"] = "other-protocol"
            else:
                content["models"]["rl_good"]["checkpoint_sha256"] = "other-checkpoint"
        (path / "summary.json").write_text(json.dumps(content))
    with pytest.raises(ValueError, match="different data protocols|changed between validation"):
        report(SimpleNamespace(runs=tmp_path / "runs", output=tmp_path / "report"))
