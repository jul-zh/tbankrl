"""One protocol for baseline, reranking and RL; evaluation never updates weights."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .baseline import popularity_evaluation, regression_metrics
from .data import FIRST_ITEM, load_manifest, load_split
from .io import sha256, write_json
from .metrics import behavior, paired_bootstrap, per_user_ranking, topk
from .models import append_actions, exact_kl
from .scaling import evaluation_identity
from .rl_schedule import rl_updates
from .training import autocast, device, freeze, load_model, seed_all, tensor_batch


@torch.no_grad()
def candidate_rewards(reward_model, tokens, candidates, chunk_size=256):
    """Score a slate in chunks; avoid a [batch * catalog * context] tensor."""
    batch, count = candidates.shape
    result = []
    owners = torch.arange(batch, device=tokens.device).repeat_interleave(count)
    flat = candidates.flatten()
    for start in range(0, len(flat), chunk_size):
        result.append(reward_model(tokens[owners[start:start + chunk_size]], flat[start:start + chunk_size]).float())
    return torch.cat(result).view(batch, count)


@torch.no_grad()
def score_policy(model, reference, reward_model, data, popularity, dev, batch_size, rerank_lambda=None, include_rollout=True, deadline=None):
    slates, rewards, observed_predictions, rollout_steps = [], [], [], []
    for start in range(0, len(data["targets"]), batch_size):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Evaluation deadline reached between batches")
        tokens, target, _ = tensor_batch(data, slice(start, start + batch_size), dev)
        with autocast(dev):
            logits = model(tokens)
            if rerank_lambda is not None:
                shortlist = logits.topk(min(100, logits.shape[-1] - FIRST_ITEM), dim=-1).indices
                predicted = candidate_rewards(reward_model, tokens, shortlist)
                scores = logits.log_softmax(dim=-1).gather(1, shortlist) + rerank_lambda * predicted
                order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :min(20, shortlist.shape[1])]
                slate = shortlist.gather(1, order)
            else:
                # Same stable tie breaking as CPU baseline.
                slate = torch.as_tensor(topk(logits.float().cpu().numpy(), 20), device=dev)
            slates.append(slate.cpu().numpy())
            rewards.append(candidate_rewards(reward_model, tokens, slate[:, :10]).mean(dim=1).cpu().numpy())
            observed_predictions.append(reward_model(tokens, target).float().cpu().numpy())
            if rerank_lambda is None and include_rollout:
                generated = []
                current = tokens
                rows = []
                for step in range(5):
                    current_logits = model(current)
                    action = torch.distributions.Categorical(logits=current_logits).sample()
                    reward = reward_model(current, action).float()
                    kl = exact_kl(current_logits, reference(current))
                    seen = tokens.eq(action[:, None]).any(dim=1)
                    repeated = torch.stack(generated, dim=1).eq(action[:, None]).any(dim=1) if generated else torch.zeros(len(tokens), dtype=torch.bool, device=dev)
                    rows.append(torch.stack([reward, kl, seen.float(), repeated.float()], dim=-1).cpu().numpy())
                    generated.append(action)
                    current = append_actions(current, action)
                rollout_steps.append(np.stack(rows, axis=1))
    recommendations = np.concatenate(slates)
    per_user = per_user_ranking(recommendations, data["targets"])
    per_user["rm_reward@10"] = np.concatenate(rewards)
    metrics = {
        "ranking": {name: float(values.mean()) for name, values in per_user.items() if name != "rm_reward@10"},
        "rm_reward@10": float(per_user["rm_reward@10"].mean()),
        "behavior@10": behavior(recommendations[:, :10], data["contexts"], popularity),
    }
    if rollout_steps:
        rollout = np.concatenate(rollout_steps)
        metrics["synthetic_rollout_5"] = [dict(zip(
            ["rm_reward", "kl", "seen_in_initial_context", "repeat_within_rollout"],
            rollout[:, step].mean(axis=0).astype(float).tolist())) for step in range(5)]
        metrics["rollout_note"] = "Sampled recommendations are appended as fictitious interactions; this is not observed user return."
    return metrics, per_user, np.concatenate(observed_predictions), recommendations


def verify_model(path, manifest_hash, dev, kind=None):
    model, checkpoint = load_model(path, dev)
    if checkpoint["run_spec"]["manifest_sha256"] != manifest_hash:
        raise ValueError(f"Checkpoint {path} belongs to another data protocol")
    if kind and checkpoint["model_spec"]["kind"] != kind:
        raise ValueError(f"Expected {kind} checkpoint: {path}")
    return freeze(model), checkpoint


def compatible_reward(path, data_dir, dev, source_data=None):
    """Explicit transfer of unchanged RM weights across nested training cohorts.

    A source manifest is required; equality of vocab size alone is insufficient.
    The original checkpoint and its training manifest are never relabeled.
    """
    model, checkpoint = load_model(path, dev)
    if checkpoint["model_spec"]["kind"] != "reward":
        raise ValueError("Expected a reward-model checkpoint")
    current = load_manifest(data_dir)
    current_hash = sha256(Path(data_dir) / "manifest.json")
    trained_hash = checkpoint["run_spec"]["manifest_sha256"]
    transfer = None
    if trained_hash != current_hash:
        if source_data is None:
            raise ValueError("Reward model has another training manifest; supply its original source data explicitly")
        source_data = Path(source_data)
        source = load_manifest(source_data)
        if sha256(source_data / "manifest.json") != trained_hash:
            raise ValueError("Reward source manifest does not match the checkpoint")
        if evaluation_identity(source_data) != evaluation_identity(data_dir):
            raise ValueError("Reward transfer requires identical token IDs, dates and evaluation arrays")
        if current["source_csv_sha256"] != source["source_csv_sha256"]:
            raise ValueError("Reward transfer source CSV differs")
        transfer = {"original_training_manifest_sha256": trained_hash,
                    "current_training_manifest_sha256": current_hash,
                    "checkpoint_sha256": sha256(path), "source_data": str(source_data),
                    "note": "Unchanged RM trained on the original cohort; only policy training uses expanded data"}
    if checkpoint["model_spec"]["kwargs"]["vocab_size"] != current["vocab_size"]:
        raise ValueError("Reward vocabulary differs")
    return freeze(model), checkpoint, transfer


@torch.no_grad()
def compare_real_and_synthetic_prefixes(model, reference, reward_model, data, dev, batch_size, seed, deadline=None):
    """Same users and first prefixes; only the source of later contexts differs."""
    users = len(data["targets"]) // 5
    if not users:
        return {"n_users": 0, "note": "No users with five eligible held-out targets"}
    contexts = data["contexts"].reshape(users, 5, -1)
    output = {"n_users": users, "note": "Survivor cohort with five eligible real prefixes; not a causal comparison. Ties/OOV can create gaps between eligible targets."}
    for mode in ("real_prefixes", "synthetic_prefixes"):
        seed_all(seed)
        batch_results = []
        for start in range(0, users, batch_size):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Prefix evaluation deadline reached between batches")
            batch = torch.as_tensor(contexts[start:start + batch_size], dtype=torch.long, device=dev)
            initial, current = batch[:, 0], batch[:, 0]
            steps = []
            for step in range(5):
                if mode == "real_prefixes":
                    current = batch[:, step]
                with autocast(dev):
                    logits = model(current)
                    action = torch.distributions.Categorical(logits=logits).sample()
                    predicted = reward_model(current, action).float()
                    kl = exact_kl(logits, reference(current))
                seen = initial.eq(action[:, None]).any(dim=1).float()
                steps.append(torch.stack([predicted, kl, seen], dim=-1).cpu().numpy())
                if mode == "synthetic_prefixes":
                    current = append_actions(current, action)
            batch_results.append(np.stack(steps, axis=1))
        means = np.concatenate(batch_results).mean(axis=0)
        output[mode] = [dict(zip(["rm_reward", "kl", "seen_in_initial_context"], row.astype(float).tolist())) for row in means]
    return output


def evaluate(args):
    data_dir, root = Path(args.data), Path(args.runs)
    manifest_hash = sha256(data_dir / "manifest.json")
    manifest = load_manifest(data_dir)
    data = load_split(data_dir, args.split)
    real_prefixes = load_split(data_dir, args.split + "_real_prefixes")
    config, dev = manifest["config"], device()
    seed_all(config["seed"])
    popularity = np.load(data_dir / "popularity.npy")
    item_rewards = np.load(data_dir / "item_rewards.npy")
    rm_path = root / args.reward_model / "best.pt"
    reward_model, _, reward_transfer = compatible_reward(rm_path, data_dir, dev, getattr(args, "reward_source_data", None))
    # The KL reference must be the same SFT policy for all models.
    reference_path = root / "sft" / "best.pt"
    reference, _ = verify_model(reference_path, manifest_hash, dev, "policy")
    output = root / "evaluation" / args.split
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "split": args.split, "manifest_sha256": manifest_hash, "n_users": len(data["users"]),
        "reward_checkpoint_sha256": sha256(rm_path), "reference_checkpoint_sha256": sha256(reference_path),
        "temporal_semantics": manifest["temporal_semantics"],
        "note": "Warm singleton targets, one per user; observed-row proxy; RM reward is a model prediction.",
        "models": {}, "paired_vs_sft": {},
        "reward_transfer": reward_transfer,
    }
    results, per_users = {}, {}
    summary["models"]["popularity"], per_users["popularity"], _ = popularity_evaluation(data, popularity)
    for name in args.names:
        if getattr(args, "progress", False):
            print(f"{args.split.capitalize()} evaluation: {name}", flush=True)
        if name == "rerank":
            model, checkpoint = reference, None
            path = reference_path
        else:
            path = root / name / "best.pt"
            model, checkpoint = verify_model(path, manifest_hash, dev, "policy")
            if checkpoint["run_spec"]["stage"] in {"rl", "sft_control"}:
                dependencies = checkpoint["run_spec"]["dependencies"]
                if dependencies != {"init_sha256": sha256(reference_path), "reward_sha256": sha256(rm_path)}:
                    raise ValueError("RL used a different SFT or reward model; comparison would be inconsistent")
        # Identical seed per policy gives a repeatable rollout protocol.
        seed_all(config["seed"])
        rollout_names = getattr(args, "rollout_names", None)
        include_rollout = rollout_names is None or name in rollout_names
        metrics, per_user, predicted, slate = score_policy(
            model, reference, reward_model, data, popularity, dev, config["eval_batch_size"],
            args.rerank_lambda if name == "rerank" else None,
            include_rollout=include_rollout,
        )
        metrics["checkpoint_sha256"] = sha256(path)
        if name != "rerank" and include_rollout:
            metrics["real_vs_synthetic_context"] = compare_real_and_synthetic_prefixes(
                model, reference, reward_model, real_prefixes, dev, config["eval_batch_size"], config["seed"])
        if name == "rerank":
            metrics["lambda"] = args.rerank_lambda
        if checkpoint is not None:
            metrics["checkpoint_step"] = checkpoint["step"]
            if "training_phase" in checkpoint:
                metrics["training_phase"] = checkpoint["training_phase"]
            metrics["stage"] = checkpoint["run_spec"]["stage"]
            settings = checkpoint["run_spec"].get("settings", {})
            if settings.get("training_schedule"):
                metrics["rl_updates"] = rl_updates(settings, checkpoint["step"])
                if not metrics["rl_updates"]:
                    metrics["stage"] = "sft_control"
            metrics["checkpoint_selection"] = checkpoint.get("selection", {
                "rule": checkpoint["run_spec"].get("selection_rule", "max_validation_ndcg")})
        summary["models"][name], per_users[name] = metrics, per_user
        results[name] = predicted
        np.savez_compressed(output / f"{name}_per_user.npz", users=data["users"], targets=data["targets"],
                            slate=slate, **per_user)
        if model is not reference:
            del model
    actual = data["rewards"]
    predictions = {
        "global_mean": np.full(len(actual), manifest["global_train_reward"]),
        "item_mean_smoothed": item_rewards[data["targets"]],
    }
    if results:
        predictions["reward_model"] = next(iter(results.values()))
    summary["reward_model_evaluation"] = {name: regression_metrics(predicted, actual)
                                          for name, predicted in predictions.items()}
    if "reward_model" in predictions:
        predicted = predictions["reward_model"]
        lengths = (data["contexts"] >= FIRST_ITEM).sum(axis=1)
        target_popularity = popularity[data["targets"]]
        train_item_median = np.median(popularity[FIRST_ITEM:])
        groups = {"history_le_5": lengths <= 5, "history_gt_5": lengths > 5,
                  "items_low_popularity": target_popularity <= train_item_median,
                  "items_high_popularity": target_popularity > train_item_median}
        summary["reward_model_groups"] = {name: regression_metrics(predicted[mask], actual[mask])
                                           for name, mask in groups.items()}
    if "sft" in per_users:
        for name, per_user in per_users.items():
            if name == "sft":
                continue
            summary["paired_vs_sft"][name] = {
                metric: paired_bootstrap(per_users["sft"][metric], values)
                for metric, values in per_user.items() if metric in per_users["sft"]
            }
    write_json(output / "summary.json", summary)
    if not getattr(args, "quiet", False):
        print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
