"""RL estimators, explicit RM transfer, recovery, and a tiny complete study."""

import copy
import itertools
import json
import os
import shutil
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from recsys.data import load_manifest
from recsys.evaluation import compatible_reward
from recsys.io import sha256, write_json
from recsys.models import Policy, RewardModel, exact_kl
from recsys.rl_objectives import leave_one_out, modified_rl_loss
from recsys.rl_study import fit, run
from recsys.rl_frontier import fraction_tag
from recsys.scaling import prepare_scaling
from recsys.training import atomic_checkpoint, freeze, read_checkpoint
from test_scaling import scaling_fixture


def objective_settings(**overrides):
    return {"beta": .3, "gamma": .95, "horizon": 1, "repeat_penalty": 0.,
            "ce_weight": 0., "baseline_mode": "ema", "group_size": 1, **overrides}


def test_loo_expected_gradient_matches_exact_objective():
    class ToyPolicy(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

        def forward(self, tokens):
            return torch.cat([self.theta.new_full((3,), -1e9), self.theta]).expand(len(tokens), -1)

    class ToyReward(torch.nn.Module):
        def forward(self, tokens, action):
            return action.eq(4).float()

    policy = ToyPolicy([.4, -.2])
    reference = freeze(ToyPolicy([-.3, .1]))
    tokens, target = torch.tensor([[1, 3, 0]]), torch.tensor([4])
    probabilities = policy.theta.softmax(dim=0)
    expected = torch.autograd.grad(-probabilities[1] + .3 * exact_kl(policy(tokens), reference(tokens)).mean(), policy.theta)[0]
    accumulated = torch.zeros_like(expected)
    for pair in itertools.product(range(2), repeat=2):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.distributions.Categorical, "sample", lambda self, pair=pair: torch.tensor(pair) + 3)
            loss, _ = modified_rl_loss(policy, reference, ToyReward(), tokens, target,
                                      objective_settings(baseline_mode="loo", group_size=2), 0.)
            gradient = torch.autograd.grad(loss, policy.theta)[0]
        accumulated += (probabilities[pair[0]] * probabilities[pair[1]]).detach() * gradient
    torch.testing.assert_close(accumulated, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(leave_one_out(torch.tensor([1., 2., 6.]), 3), torch.tensor([4., 3.5, 1.5]))


@pytest.mark.parametrize("horizon", [1, 5])
def test_hybrid_loo_updates_policy_and_freezes_rm_and_reference(horizon):
    torch.set_num_threads(1)
    policy = Policy(9, 5, hidden=16, layers=1, heads=2, dropout=0.).eval()
    reference = freeze(copy.deepcopy(policy))
    reward = freeze(RewardModel(9, hidden=8))
    before = {k: v.clone() for k, v in reference.state_dict().items()}
    reward_before = {k: v.clone() for k, v in reward.state_dict().items()}
    tokens, target = torch.tensor([[1, 3, 0, 0, 0], [1, 4, 5, 0, 0]]), torch.tensor([4, 6])
    loss, stats = modified_rl_loss(policy, reference, reward, tokens, target,
        objective_settings(horizon=horizon, baseline_mode="loo", group_size=4, ce_weight=.1, repeat_penalty=.2), 0.)
    loss.backward()
    assert stats["anchor_ce"] > 0
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
    torch.optim.AdamW(policy.parameters(), lr=.001).step()
    assert any(not torch.equal(v, policy.state_dict()[k]) for k, v in before.items())
    for k, v in before.items(): torch.testing.assert_close(v, reference.state_dict()[k], atol=0, rtol=0)
    for k, v in reward_before.items(): torch.testing.assert_close(v, reward.state_dict()[k], atol=0, rtol=0)
    assert all(p.grad is None for p in reference.parameters())
    assert all(p.grad is None for p in reward.parameters())


def test_disabled_reinforce_matches_ce_and_exact_kl_gradients():
    from torch.nn import functional as F
    torch.set_num_threads(1)
    torch.manual_seed(123)
    policy = Policy(9, 5, hidden=16, layers=1, heads=2, dropout=0.).eval()
    reference = freeze(copy.deepcopy(policy))
    with torch.no_grad():
        next(policy.parameters()).add_(.02 * torch.randn_like(next(policy.parameters())))
    reward = freeze(RewardModel(9, hidden=8))
    tokens = torch.tensor([[1, 3, 0, 0, 0], [1, 4, 5, 0, 0]])
    targets = torch.tensor([4, 6])
    parameters = list(policy.parameters())
    expected = .3 * F.cross_entropy(policy(tokens), targets) + .3 * exact_kl(policy(tokens), reference(tokens)).mean()
    expected_grads = torch.autograd.grad(expected, parameters)
    for seed, baseline in [(42, 0.), (123, 100.)]:
        torch.manual_seed(seed)
        loss, stats = modified_rl_loss(policy, reference, reward, tokens, targets,
                                      objective_settings(ce_weight=.3, reinforce_weight=0.), baseline)
        torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-5)
        for actual, wanted in zip(torch.autograd.grad(loss, parameters), expected_grads):
            torch.testing.assert_close(actual, wanted, atol=1e-6, rtol=1e-5)
        assert stats["reinforce_weight"] == 0
    with pytest.raises(ValueError, match="H=1"):
        modified_rl_loss(policy, reference, reward, tokens, targets,
                         objective_settings(horizon=5, reinforce_weight=0.), 0.)


@pytest.mark.parametrize("item_only", [False, True])
def test_catalog_reward_matches_individual_scoring(item_only):
    torch.manual_seed(42)
    reward = freeze(RewardModel(9, hidden=8, item_only=item_only))
    tokens = torch.tensor([[1, 3, 2, 0], [1, 4, 5, 0]])
    direct = torch.stack([reward(tokens, torch.full((2,), item)) for item in range(3, 9)], dim=1)
    for chunk in [1, 2, 100]:
        torch.testing.assert_close(reward.score_catalog(tokens, chunk), direct, atol=1e-7, rtol=1e-6)
    assert not reward.score_catalog(tokens).requires_grad


def test_full_catalog_gradient_matches_closed_form_without_sampling(monkeypatch):
    class ToyPolicy(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

        def forward(self, tokens):
            return torch.cat([self.theta.new_full((3,), -1e9), self.theta]).expand(len(tokens), -1)

    class ToyReward(torch.nn.Module):
        def score_catalog(self, tokens, chunk_size):
            return torch.tensor([[.2, .8]]).expand(len(tokens), -1)

    def forbid_sampling(*args, **kwargs):
        raise AssertionError("The full-catalog estimator must not sample an action")

    monkeypatch.setattr(torch.distributions.Categorical, "sample", forbid_sampling)
    policy = ToyPolicy([.4, -.2])
    reference = freeze(ToyPolicy([-.3, .1]))
    tokens, targets = torch.tensor([[1, 3, 0]]), torch.tensor([4])
    p = policy.theta.detach().softmax(0)
    q = reference.theta.detach().softmax(0)
    reward = torch.tensor([.2, .8])
    log_ratio = p.log() - q.log()
    expected = p * ((p * reward).sum() - reward) + .3 * p * (log_ratio - (p * log_ratio).sum())
    expected += .3 * (p - torch.tensor([0., 1.]))
    loss, stats = modified_rl_loss(policy, reference, ToyReward(), tokens, targets,
        objective_settings(objective_mode="full_catalog", ce_weight=.3), 999.)
    actual = torch.autograd.grad(loss, policy.theta)[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert stats["full_catalog_items"] == 2 and not stats["action_sampling"]
    with pytest.raises(ValueError, match="H=1"):
        modified_rl_loss(policy, reference, ToyReward(), tokens, targets,
            objective_settings(objective_mode="full_catalog", horizon=5), 0.)


def study_fixture(tmp_path):
    csv_path, source = scaling_fixture(tmp_path)
    data = tmp_path / "scaled"
    prepare_scaling(csv_path, source, data, {"max_users": 7, "max_added_examples": 1000})
    manifest = load_manifest(data / "large")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    kwargs = {"vocab_size": manifest["vocab_size"], "context": 6, "hidden": 16, "layers": 1, "heads": 2, "dropout": .1}
    atomic_checkpoint(inputs / "sft.pt", {"model": Policy(**kwargs).state_dict(), "model_spec": {"kind": "policy", "kwargs": kwargs},
        "step": 0, "run_spec": {"stage": "sft", "manifest_sha256": sha256(data / "large/manifest.json")}})
    reward_kwargs = {"vocab_size": manifest["vocab_size"], "hidden": 8}
    atomic_checkpoint(inputs / "rm.pt", {"model": RewardModel(**reward_kwargs).state_dict(),
        "model_spec": {"kind": "reward", "kwargs": reward_kwargs}, "step": 0,
        "run_spec": {"stage": "rm", "manifest_sha256": sha256(source / "manifest.json")}})
    plan = {"seed": 42, "overall_minutes": 3, "evaluation_reserve_minutes": .01, "checkpoint_minutes": 0,
            "relevance_fraction": .95, "rerank_lambdas": [.5, 1., 2.],
            "training": {**objective_settings(), "batch_size": 4, "learning_rate": .0001,
                         "max_steps": 2, "max_minutes": 1, "eval_every": 1},
            "branches": [{"name": "rl_test"}, {"name": "rl_loo", "baseline_mode": "loo", "group_size": 2, "ce_weight": .1}]}
    config = tmp_path / "plan.json"
    write_json(config, plan)
    args = SimpleNamespace(data=str(data / "large"), runs=str(tmp_path / "runs"), backup=str(tmp_path / "backup"),
        source_sft=str(inputs / "sft.pt"), source_rm=str(inputs / "rm.pt"), reward_source_data=str(source),
        config=str(config), stage="prepare", allow_cpu=True)
    return args, plan


def test_transfer_and_full_rl_study(tmp_path, monkeypatch):
    import recsys.rl_study as study
    torch.set_num_threads(1)
    monkeypatch.setattr(study, "device", lambda: torch.device("cpu"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    args, plan = study_fixture(tmp_path)
    source_hash = sha256(args.source_rm)
    with pytest.raises(ValueError, match="source data explicitly"):
        compatible_reward(args.source_rm, args.data, torch.device("cpu"))
    _, state, transfer = compatible_reward(args.source_rm, args.data, torch.device("cpu"), args.reward_source_data)
    assert transfer["original_training_manifest_sha256"] == state["run_spec"]["manifest_sha256"]
    run(args)
    args.stage = "pilot"
    run(args)
    args.stage = "train"
    run(args)
    root, backup = Path(args.runs), Path(args.backup)
    journal = json.loads((root / "experiment.json").read_text())
    assert journal["completed"] == ["rl_test", "rl_loo"]
    assert journal["pilot_completed"] == ["pilot_h1", "pilot_loo", "pilot_h5"]
    result = json.loads((root / "evaluation/validation/summary.json").read_text())
    assert {"sft", "rerank", "rerank_0.5", "rerank_2", "rl_test", "rl_test_last", "rl_loo"} <= result["models"].keys()
    assert result["reward_transfer"] is not None
    assert sha256(root / "rm/best.pt") == sha256(args.source_rm) == source_hash
    assert (backup / "branches/rl_test/last.pt").exists()
    assert not (root / "evaluation/test").exists()
    before = (root / "branches/rl_test/last.pt").stat().st_mtime_ns
    run(args)
    assert (root / "branches/rl_test/last.pt").stat().st_mtime_ns == before
    assert json.loads((root / "experiment.json").read_text())["deadline"] == journal["deadline"]
    # A same-sized but permuted vocabulary must be rejected, even with an internally valid source manifest.
    swapped = tmp_path / "swapped"
    shutil.copytree(args.reward_source_data, swapped)
    mapping = json.loads((swapped / "vocab.json").read_text())
    a, b = list(mapping)[:2]
    mapping[a], mapping[b] = mapping[b], mapping[a]
    write_json(swapped / "vocab.json", mapping)
    m = json.loads((swapped / "manifest.json").read_text())
    m["file_sha256"]["vocab.json"] = sha256(swapped / "vocab.json")
    write_json(swapped / "manifest.json", m)
    state["run_spec"]["manifest_sha256"] = sha256(swapped / "manifest.json")
    atomic_checkpoint(tmp_path / "wrong_rm.pt", state)
    with pytest.raises(ValueError, match="identical token IDs"):
        compatible_reward(tmp_path / "wrong_rm.pt", args.data, torch.device("cpu"), swapped)


def test_rl_resume_after_sigint_matches_full_updates(tmp_path, monkeypatch):
    import recsys.rl_study as study
    torch.set_num_threads(1)
    monkeypatch.setattr(study, "device", lambda: torch.device("cpu"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    args, plan = study_fixture(tmp_path)
    run(args)
    journal = json.loads((Path(args.runs) / "experiment.json").read_text())
    full_settings = {**plan["training"], "name": "full"}
    fit(args, full_settings, plan, journal)
    interrupted = False
    original = study.update

    def stop_after_update(*a, **kw):
        nonlocal interrupted
        success = original(*a, **kw)
        if success and not interrupted:
            interrupted = True
            signal.raise_signal(signal.SIGINT)
        return success

    monkeypatch.setattr(study, "update", stop_after_update)
    with pytest.raises(KeyboardInterrupt):
        fit(args, {**full_settings, "name": "resumed"}, plan, journal)
    monkeypatch.setattr(study, "update", original)
    fit(args, {**full_settings, "name": "resumed"}, plan, journal)
    a = read_checkpoint(Path(args.runs) / "branches/full/last.pt", torch.device("cpu"))
    b = read_checkpoint(Path(args.runs) / "branches/resumed/last.pt", torch.device("cpu"))
    assert a["step"] == b["step"] == 2
    for key in a["model"]: torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)


def test_multiple_constraints_and_branch_seeds(tmp_path, monkeypatch):
    import recsys.rl_study as study
    torch.set_num_threads(1)
    monkeypatch.setattr(study, "device", lambda: torch.device("cpu"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    args, plan = study_fixture(tmp_path)
    plan.update(relevance_fraction=1., selection_fractions=[1., .99, .98, .95],
                branches=[{"name": "control_s42", "family": "control", "seed": 42},
                          {"name": "anchor_s42", "family": "anchor", "seed": 42, "ce_weight": .3, "reinforce_weight": 0.},
                          {"name": "control_s123", "family": "control", "seed": 123},
                          {"name": "exact_s42", "family": "exact", "seed": 42, "ce_weight": .3, "objective_mode": "full_catalog"}])
    write_json(args.config, plan)
    run(args)
    args.stage = "train"
    run(args)
    root = Path(args.runs)
    reference = json.loads((root / "reference.json").read_text())
    for branch in plan["branches"]:
        folder = root / "branches" / branch["name"]
        spec = json.loads((folder / "run.json").read_text())
        assert spec["seed"] == branch["seed"]
        assert spec["stage"] == ("sft_control" if branch.get("reinforce_weight", 1.) == 0 else "rl")
        for fraction in plan["selection_fractions"]:
            state = read_checkpoint(folder / ("best_" + fraction_tag(fraction) + ".pt"), torch.device("cpu"))
            assert state["selection"]["relevance_fraction"] == fraction
            assert state["selection"]["metrics"]["ndcg@10"] >= fraction * reference["ndcg@10"]
    summary = json.loads((root / "evaluation/validation/summary.json").read_text())
    assert len(summary["selection_by_constraint"]) == 4
    primary = json.loads((root / "reports/selection.json").read_text())
    assert primary["validation_ndcg_floor"] == summary["models"]["sft"]["ranking"]["ndcg@10"]
    aggregate = json.loads((root / "reports/frontier.json").read_text())
    assert len(aggregate["points"]) == 16
    assert {x["n_seeds"] for x in aggregate["seed_aggregates"]} == {1, 2}
    a = read_checkpoint(root / "branches/control_s42/last.pt", torch.device("cpu"))
    b = read_checkpoint(root / "branches/control_s123/last.pt", torch.device("cpu"))
    assert any(not torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert not (root / "evaluation/test").exists()


def test_equal_update_schedules_reset_all_optimizers_once(tmp_path, monkeypatch):
    import recsys.rl_study as study
    torch.set_num_threads(1)
    monkeypatch.setattr(study, "device", lambda: torch.device("cpu"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    args, plan = study_fixture(tmp_path)
    plan["training"].update(max_steps=4, ce_weight=.3, switch_step=2, snapshot_every=2)
    plan["branches"] = [
        {"name": "warm", "training_schedule": "ce_then_rl"},
        {"name": "reverse", "training_schedule": "rl_then_ce"},
        {"name": "joint", "training_schedule": "rl_then_rl"},
        {"name": "ce", "training_schedule": "ce_then_ce", "reinforce_weight": 0.},
    ]
    write_json(args.config, plan)
    run(args)
    args.stage = "train"
    run(args)
    result = json.loads((Path(args.runs) / "evaluation/validation/summary.json").read_text())
    assert result["models"]["warm_last"]["training_phase"] == 2
    assert result["models"]["warm_last"]["rl_updates"] == 2
    for branch in plan["branches"]:
        folder = Path(args.runs) / "branches" / branch["name"]
        state = read_checkpoint(folder / "last.pt", torch.device("cpu"))
        assert state["step"] == 4 and state["training_phase"] == 2 and state["optimizer_resets"] == 1
        assert state["optimizer"]["state"]
        assert all(int(v["step"]) == 2 for v in state["optimizer"]["state"].values())
        assert (folder / "step_0000002.pt").exists() and (folder / "step_0000004.pt").exists()
        assert not (folder / "step_0000001.pt").exists()
        history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
        flags = {h["step"]: h["reinforce_weight"] for h in history if h["step"]}
        expected = {"warm": (0., 1.), "reverse": (1., 0.), "joint": (1., 1.), "ce": (0., 0.)}[branch["name"]]
        assert flags[1] == flags[2] == expected[0] and flags[3] == flags[4] == expected[1]


def test_schedule_resume_exactly_at_reset_boundary(tmp_path, monkeypatch):
    import recsys.rl_study as study
    torch.set_num_threads(1)
    monkeypatch.setattr(study, "device", lambda: torch.device("cpu"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    args, plan = study_fixture(tmp_path)
    run(args)
    journal = json.loads((Path(args.runs) / "experiment.json").read_text())
    config = {**plan["training"], "name": "full_schedule", "ce_weight": .3,
              "max_steps": 4, "switch_step": 2, "training_schedule": "ce_then_rl"}
    fit(args, config, plan, journal)
    original, updates = study.update, 0

    def stop_at_boundary(*a, **kw):
        nonlocal updates
        success = original(*a, **kw)
        updates += int(success)
        if updates == 2:
            signal.raise_signal(signal.SIGINT)
        return success

    monkeypatch.setattr(study, "update", stop_at_boundary)
    with pytest.raises(KeyboardInterrupt):
        fit(args, {**config, "name": "resumed_schedule"}, plan, journal)
    middle = read_checkpoint(Path(args.runs) / "branches/resumed_schedule/last.pt", torch.device("cpu"))
    assert middle["step"] == 2 and middle["training_phase"] == 1 and middle["optimizer_resets"] == 0
    monkeypatch.setattr(study, "update", original)
    fit(args, {**config, "name": "resumed_schedule"}, plan, journal)
    full = read_checkpoint(Path(args.runs) / "branches/full_schedule/last.pt", torch.device("cpu"))
    resumed = read_checkpoint(Path(args.runs) / "branches/resumed_schedule/last.pt", torch.device("cpu"))
    assert resumed["optimizer_resets"] == full["optimizer_resets"] == 1
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], atol=0, rtol=0)
