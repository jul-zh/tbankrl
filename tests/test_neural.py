"""Run in Colab with neural dependencies; deliberately tiny CPU workloads."""

import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from recsys.models import Policy, RewardModel, append_actions, exact_kl, rl_loss
from recsys.training import atomic_checkpoint, freeze, load_model

torch.set_num_threads(1)


def tiny_policy():
    return Policy(vocab_size=9, context=5, hidden=16, layers=1, heads=2, dropout=0.)


def test_padding_does_not_change_prediction():
    model = tiny_policy().eval()
    torch.testing.assert_close(model(torch.tensor([[1, 3, 4, 0, 0]])), model(torch.tensor([[1, 3, 4]])), atol=1e-6, rtol=1e-5)


def test_gpt_causal_mask_blocks_future_tokens():
    model = tiny_policy().eval()
    a = model.backbone(torch.tensor([[1, 3, 4]]), use_cache=False).last_hidden_state
    b = model.backbone(torch.tensor([[1, 3, 8]]), use_cache=False).last_hidden_state
    torch.testing.assert_close(a[:, :2], b[:, :2], atol=1e-6, rtol=1e-5)


def test_context_append_preserves_padding_and_slides_full_context():
    original = torch.tensor([[1, 3, 0], [1, 4, 5]])
    assert append_actions(original, torch.tensor([6, 7])).tolist() == [[1, 3, 6], [4, 5, 7]]
    assert original.tolist() == [[1, 3, 0], [1, 4, 5]]


def test_sft_loss_uses_next_target_and_has_finite_gradients():
    model = tiny_policy()
    tokens = torch.tensor([[1, 3, 0, 0, 0], [1, 4, 5, 0, 0]])
    targets = torch.tensor([4, 6])
    logits = model(tokens)
    assert (logits.argmax(dim=1) >= 3).all()
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_exact_kl_matches_distribution_and_has_correct_gradient():
    a = torch.tensor([[-1e9, -1e9, -1e9, 0., 1.]], requires_grad=True)
    b = torch.tensor([[-1e9, -1e9, -1e9, 1., 0.]])
    actual = exact_kl(a, b)
    expected = torch.distributions.kl_divergence(torch.distributions.Categorical(logits=a[:, 3:]), torch.distributions.Categorical(logits=b[:, 3:]))
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert a.grad[0, 4] > 0 and a.grad[0, 3] < 0


@pytest.mark.parametrize("horizon", [1, 5])
def test_rl_updates_only_policy(horizon):
    torch.manual_seed(42)
    policy = tiny_policy().eval()
    reference = freeze(copy.deepcopy(policy))
    reward = freeze(RewardModel(9, hidden=8))
    tokens = torch.tensor([[1, 3, 0, 0, 0], [1, 4, 5, 0, 0]])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=.001)
    before = policy.backbone.wte.weight.detach().clone()
    loss, stats = rl_loss(policy, reference, reward, tokens, .03, .2, horizon, .95, 0.)
    assert np.isfinite(stats["objective_return"])
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, policy.backbone.wte.weight)
    assert all(p.grad is None for p in reference.parameters())
    assert all(p.grad is None for p in reward.parameters())


def test_reward_model_depends_on_history_and_candidate():
    model = RewardModel(9, hidden=8)
    value = model(torch.tensor([[1, 3, 0], [1, 4, 0]]), torch.tensor([5, 5]))
    assert ((value > 0) & (value < 1)).all()
    assert not torch.equal(value[0], value[1])


def test_inference_checkpoint_roundtrip(tmp_path):
    model = tiny_policy().eval()
    path = tmp_path / "model.pt"
    spec = {"kind": "policy", "kwargs": {"vocab_size": 9, "context": 5, "hidden": 16, "layers": 1, "heads": 2, "dropout": 0.}}
    atomic_checkpoint(path, {"model_spec": spec, "model": model.state_dict()})
    loaded, _ = load_model(path, torch.device("cpu"))
    tokens = torch.tensor([[1, 3, 0, 0, 0]])
    torch.testing.assert_close(model(tokens), loaded(tokens))


def test_training_resume_matches_uninterrupted_steps_and_full_pipeline(tmp_path, monkeypatch):
    import pandas as pd
    import recsys.training as training
    from recsys.cli import main
    from recsys.data import prepare

    source = tmp_path / "interactions.csv"
    rows = [(u, 10 + (u + day) % 5, f"2021-01-{day:02d}", 20 + 10 * ((u + day) % 5))
            for u in range(8) for day in range(1, 21)]
    pd.DataFrame(rows, columns=["user_id", "item_id", "last_watch_dt", "watched_pct"]).to_csv(source, index=False)
    config = {"seed": 42, "context": 5, "min_train_events": 3, "max_users": 8,
              "max_train_examples": 200, "max_eval_users": 8,
              "train_fraction": .5, "validation_fraction": .25,
              "model": {"hidden": 16, "layers": 1, "heads": 2, "dropout": .1},
              "checkpoint_minutes": 0, "eval_batch_size": 8, "validation_train_limit": 8}
    for stage in ["sft", "rm", "rl"]:
        config[stage] = {"batch_size": 4, "learning_rate": .001,
                         "max_steps": 2, "max_minutes": 2, "eval_every": 1, "gamma": .95}
    data, runs = tmp_path / "prepared", tmp_path / "runs"
    prepare(source, data, config)
    common = ["--data", str(data), "--runs", str(runs)]
    main(["train", "--stage", "sft", "--name", "full", *common])
    original = training.atomic_checkpoint
    interrupted = False

    def interrupt_after_safe_checkpoint(path, state):
        nonlocal interrupted
        original(path, state)
        if path.name == "last.pt" and state["step"] == 1 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("simulated Colab interruption at an update boundary")

    monkeypatch.setattr(training, "atomic_checkpoint", interrupt_after_safe_checkpoint)
    with pytest.raises(KeyboardInterrupt):
        main(["train", "--stage", "sft", *common])
    monkeypatch.setattr(training, "atomic_checkpoint", original)
    main(["train", "--stage", "sft", "--resume", *common])
    full = training.read_checkpoint(runs / "full" / "last.pt", torch.device("cpu"))
    resumed = training.read_checkpoint(runs / "sft" / "last.pt", torch.device("cpu"))
    assert full["step"] == resumed["step"] == 2
    for name in full["model"]:
        torch.testing.assert_close(full["model"][name], resumed["model"][name], atol=0, rtol=0)
    main(["train", "--stage", "rm", *common])
    main(["train", "--stage", "rl", "--name", "rl_smoke", "--horizon", "5", *common])
    main(["evaluate", "--names", "sft", "rerank", "rl_smoke", *common])
    summary = json.loads((runs / "evaluation" / "validation" / "summary.json").read_text())
    assert set(summary["models"]) == {"popularity", "sft", "rerank", "rl_smoke"}
    assert len(summary["models"]["rl_smoke"]["synthetic_rollout_5"]) == 5
    assert summary["reward_model_evaluation"]["reward_model"]["n"] == 8
    comparison = summary["models"]["rl_smoke"]["real_vs_synthetic_context"]
    assert comparison["n_users"] == 8
    assert comparison["real_prefixes"][0] == comparison["synthetic_prefixes"][0]
