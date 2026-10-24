"""Longer SFT/RL continuation with a persistent whole-experiment deadline.

Run this in Colab after the first experiment. Source checkpoints and the data
manifest are immutable. New optimizer/scheduler states start at step zero;
the SFT warm start is the old validation-selected checkpoint. RL branches
start from the new selected SFT. Reward model stays identical to the old run.
"""

import argparse
import copy
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from .budget import eligible_checkpoint, evaluation_due, exclusive_run, may_train, new_deadline, remaining_seconds
from .data import load_manifest, load_split
from .io import sha256, write_json
from .metrics import per_user_ranking, topk
from .models import rl_loss
from .training import (
    atomic_checkpoint, autocast, device, freeze, load_model,
    read_checkpoint, seed_all, tensor_batch,
)


def mirror(files, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for source in files:
        source = Path(source)
        if source.is_file():
            temporary = destination / (source.name + ".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(destination / source.name)


@torch.no_grad()
def measure(policy, reward_model, data, dev, batch_size):
    """Deterministic selection on the complete fixed validation cohort."""
    policy.eval()
    ndcg, predicted_reward = [], []
    for start in range(0, len(data["targets"]), batch_size):
        tokens, targets, _ = tensor_batch(data, slice(start, start + batch_size), dev)
        with autocast(dev):
            logits = policy(tokens)
            slate_np = topk(logits.float().cpu().numpy(), 10)
            slate = torch.as_tensor(slate_np, device=dev)
            k, length = slate.shape[1], tokens.shape[1]
            expanded = tokens[:, None, :].expand(-1, k, -1).reshape(-1, length)
            rewards = reward_model(expanded, slate.reshape(-1)).float().reshape(len(tokens), k)
        ndcg.extend(per_user_ranking(slate_np, targets.cpu().numpy(), (10,))["ndcg@10"].tolist())
        predicted_reward.extend(rewards.mean(dim=1).cpu().tolist())
    return {"ndcg@10": float(np.mean(ndcg)), "rm_reward@10": float(np.mean(predicted_reward))}


def update(loss, model, optimizer, scaler):
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients:
        raise RuntimeError("No policy gradients")
    if not torch.stack([torch.isfinite(g).all() for g in gradients]).all():
        if not scaler.is_enabled():
            raise FloatingPointError("Non-finite gradients without AMP")
        previous = scaler.get_scale()
        scaler.step(optimizer)  # unscale_ recorded overflow, so this skips the update
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        print(f"AMP overflow: scale {previous:g} -> {scaler.get_scale():g}", flush=True)
        return False
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite gradient norm")
    scaler.step(optimizer)
    scaler.update()
    return True


def fit(stage, settings, source_checkpoint, reference_checkpoint, reward_checkpoint,
        data_dir, root, backup, plan, deadline):
    name = settings["name"]
    output = root / name
    output.mkdir(parents=True, exist_ok=True)
    destination = backup / name
    dev = device()
    seed_all(plan["seed"])
    rng = np.random.default_rng(plan["seed"])
    manifest_hash = sha256(data_dir / "manifest.json")
    model, initial = load_model(source_checkpoint, dev)
    reward_model, reward_state = load_model(reward_checkpoint, dev)
    freeze(reward_model)
    if initial["model_spec"]["kind"] != "policy" or reward_state["model_spec"]["kind"] != "reward":
        raise ValueError("Expected a policy warm start and a reward model")
    reference = None
    for checkpoint in (initial, reward_state):
        if checkpoint["run_spec"]["manifest_sha256"] != manifest_hash:
            raise ValueError("Source checkpoint and dataset protocol differ")
    dependencies = {"warm_start_sha256": sha256(source_checkpoint)}
    if stage == "rl":
        reference, reference_state = load_model(reference_checkpoint, dev)
        if reference_state["run_spec"]["manifest_sha256"] != manifest_hash:
            raise ValueError("Reference and dataset protocol differ")
        freeze(reference)
        dependencies = {"init_sha256": sha256(reference_checkpoint), "reward_sha256": sha256(reward_checkpoint)}
    model.requires_grad_(True)
    spec = initial["model_spec"]
    train_data = load_split(data_dir, "train")
    validation = load_split(data_dir, "validation")
    batch_size = load_manifest(data_dir)["config"]["eval_batch_size"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["max_steps"], eta_min=settings["min_learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    run_spec = {
        "stage": stage, "manifest_sha256": manifest_hash, "settings": settings,
        "seed": plan["seed"], "dependencies": dependencies,
        "beta": settings.get("beta"), "horizon": settings.get("horizon"),
        "repeat_penalty": settings.get("repeat_penalty"),
        "selection_rule": "full_validation_ndcg" if stage == "sft" else "full_validation_reward_with_ndcg_floor",
        "relevance_fraction": plan["relevance_fraction"],
        "warm_start_step": initial["step"], "optimizer_restart": True,
    }
    step, elapsed_before, baseline = 0, 0., 0.
    reference_metrics = measure(reference if reference is not None else model, reward_model, validation, dev, batch_size)
    best_metrics = dict(reference_metrics)
    last_path = output / "last.pt"
    if last_path.exists():
        saved = read_checkpoint(last_path, dev)
        if saved["run_spec"] != run_spec:
            raise ValueError(f"Cannot resume {name}: parameters, warm start or data changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        step, elapsed_before, baseline = saved["step"], saved["elapsed_seconds"], saved["baseline"]
        best_metrics = saved["best_metrics"]
        rng.bit_generator.state = saved["sampler_rng"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if dev.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])
    step_cap = settings["max_steps"]
    limits_path = root / "runtime_limits.json"
    if limits_path.exists():
        limit = json.loads(limits_path.read_text()).get(name, step_cap)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("Runtime step limits must be positive integers")
        step_cap = min(step_cap, limit)
    print(f"{name}: effective_max_steps={step_cap}, restored_step={step}", flush=True)
    started, last_save = time.monotonic(), time.monotonic()
    last_progress = started
    history_path = output / "history.jsonl"
    write_json(output / "run.json", {**run_spec, "model_spec": spec, "reference_metrics": reference_metrics,
                                    "effective_max_steps": step_cap})
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def elapsed():
        return elapsed_before + time.monotonic() - started

    def compact(metrics, rule):
        return {
            "model": model.state_dict(), "model_spec": spec, "run_spec": run_spec, "step": step,
            "validation_score": metrics["ndcg@10"] if stage == "sft" else metrics["rm_reward@10"],
            "selection": {"rule": rule, "metrics": metrics,
                          "reference_ndcg@10": reference_metrics["ndcg@10"],
                          "relevance_fraction": plan["relevance_fraction"]},
        }

    def save():
        atomic_checkpoint(last_path, {
            "model": model.state_dict(), "model_spec": spec, "run_spec": run_spec,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "step": step, "elapsed_seconds": elapsed(), "baseline": baseline, "best_metrics": best_metrics,
            "sampler_rng": copy.deepcopy(rng.bit_generator.state),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if dev.type == "cuda" else [],
        })
        mirror([last_path, output / "best.pt", output / "run.json", history_path], destination)

    def assess(loss=None, stats=None):
        nonlocal best_metrics
        metrics = measure(model, reward_model, validation, dev, batch_size)
        entry = {"step": step, "elapsed_seconds": elapsed(), "learning_rate": optimizer.param_groups[0]["lr"],
                 "validation_ndcg@10": metrics["ndcg@10"], "validation_reward@10": metrics["rm_reward@10"], **(stats or {})}
        if loss is not None:
            entry["loss"] = float(loss.detach())
        entry["feasible"] = metrics["ndcg@10"] >= plan["relevance_fraction"] * reference_metrics["ndcg@10"]
        snapshot = output / f"step_{step:07d}.pt"
        atomic_checkpoint(snapshot, compact(metrics, "intermediate_snapshot"))
        if not (output / "best.pt").exists() or eligible_checkpoint(
                stage, metrics, best_metrics, reference_metrics["ndcg@10"], plan["relevance_fraction"]):
            best_metrics = metrics
            atomic_checkpoint(output / "best.pt", compact(metrics, run_spec["selection_rule"]))
            entry["selected_best"] = True
        with history_path.open("a") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
        print(name, json.dumps(entry, allow_nan=False), flush=True)
        mirror([snapshot, output / "best.pt", history_path], destination)

    try:
        if not (output / "best.pt").exists():
            assess()
            save()
        while step < step_cap and may_train(
                deadline, plan["evaluation_reserve_minutes"], elapsed(), settings["max_minutes"]):
            model.train(stage == "sft")
            indices = rng.integers(0, len(train_data["targets"]), size=settings["batch_size"])
            tokens, targets, _ = tensor_batch(train_data, indices, dev)
            optimizer.zero_grad(set_to_none=True)
            stats = {}
            with autocast(dev):
                if stage == "sft":
                    loss = F.cross_entropy(model(tokens), targets)
                else:
                    loss, stats = rl_loss(model, reference, reward_model, tokens, settings["beta"],
                                          settings["repeat_penalty"], settings["horizon"], settings["gamma"], baseline)
            if not update(loss, model, optimizer, scaler):
                continue
            scheduler.step()
            if stage == "rl":
                baseline = .95 * baseline + .05 * stats["objective_return"]
            step += 1
            if evaluation_due(step, settings["eval_every"]):
                assess(loss, stats)
            elif step % 100 == 0:
                entry = {"step": step, "elapsed_seconds": elapsed(), "loss": float(loss.detach()), **stats}
                with history_path.open("a") as stream:
                    stream.write(json.dumps(entry, allow_nan=False) + "\n")
            if time.monotonic() - last_progress >= 30:
                print(f"{name}: step={step}, minutes={elapsed() / 60:.1f}, loss={float(loss.detach()):.4f}, "
                      f"global_remaining={remaining_seconds(deadline) / 60:.1f} min", flush=True)
                last_progress = time.monotonic()
            if time.monotonic() - last_save >= plan["checkpoint_minutes"] * 60:
                save()
                last_save = time.monotonic()
        # Include the final state even when a time limit lands between scheduled checks.
        if remaining_seconds(deadline) > 60 and not (output / f"step_{step:07d}.pt").exists():
            assess()
    finally:
        save()
        write_json(output / "timing.json", {
            "completed_steps": step, "elapsed_seconds": elapsed(),
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if dev.type == "cuda" else 0,
            "stopped_by_steps": step >= step_cap, "effective_max_steps": step_cap,
            "stopped_by_stage_time": elapsed() >= settings["max_minutes"] * 60,
            "stopped_by_global_budget": remaining_seconds(deadline) <= plan["evaluation_reserve_minutes"] * 60,
            "best_metrics": best_metrics,
        })
        mirror([output / "timing.json"], destination)


def run(args):
    data_dir, root, source, backup = map(Path, (args.data, args.runs, args.source_runs, args.backup))
    plan = json.loads(Path(args.config).read_text())
    if root.resolve() == source.resolve() or backup.resolve() == source.resolve():
        raise ValueError("Extended runs must use a separate directory from the original experiment")
    if not (root / "experiment.json").exists() and (backup / "experiment.json").exists():
        shutil.copytree(backup, root, dirs_exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    backup.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(data_dir)
    source_sft, source_rm = source / "sft/best.pt", source / "rm/best.pt"
    identity = {"manifest_sha256": sha256(data_dir / "manifest.json"), "plan": plan,
                "source_sft_sha256": sha256(source_sft), "source_rm_sha256": sha256(source_rm)}
    journal_path = root / "experiment.json"
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        if any(journal[key] != value for key, value in identity.items()):
            raise ValueError("Existing extended experiment uses different inputs or settings")
    else:
        journal = {**identity, "started_at": time.time(), "deadline": new_deadline(plan["overall_minutes"]), "completed": []}
        write_json(journal_path, journal)
        mirror([journal_path], backup)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("Select a GPU runtime; extended training is not intended for CPU")
    if remaining_seconds(journal["deadline"]) <= 0:
        print("Experiment deadline has expired. Saved results remain available.", flush=True)
        return
    mirror([root / "runtime_limits.json"], backup)
    print(f"Examples: {manifest['examples']}; remaining budget {remaining_seconds(journal['deadline']) / 60:.1f} min", flush=True)
    # Keep the exact old RM bytes, so dependency hashes remain comparable.
    (root / "rm").mkdir(exist_ok=True)
    if (root / "rm/best.pt").exists() and sha256(root / "rm/best.pt") != sha256(source_rm):
        raise ValueError("Reward model changed in the extended experiment")
    shutil.copy2(source_rm, root / "rm/best.pt")
    mirror([root / "rm/best.pt"], backup / "rm")
    stages = [("sft", plan["sft"])] + [("rl", {**plan["rl_defaults"], **branch}) for branch in plan["rl"]]
    for stage, settings in stages:
        if settings["name"] in journal["completed"]:
            continue
        if remaining_seconds(journal["deadline"]) <= (plan["evaluation_reserve_minutes"] + 1) * 60:
            break
        initial = source_sft if stage == "sft" else root / "sft/best.pt"
        fit(stage, settings, initial, root / "sft/best.pt", root / "rm/best.pt",
            data_dir, root, backup, plan, journal["deadline"])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        journal["completed"].append(settings["name"])
        write_json(journal_path, journal)
        mirror([journal_path], backup)
    # Evaluate selected and last states side by side; originals stay untouched.
    names = ["sft", "sft_original", "rerank"] if (root / "sft/best.pt").exists() else []
    if names:
        (root / "sft_original").mkdir(exist_ok=True)
        shutil.copy2(source_sft, root / "sft_original/best.pt")
    for name in ["sft"] + [branch["name"] for branch in plan["rl"]]:
        folder = root / name
        if not (folder / "best.pt").exists():
            continue
        if name != "sft":
            names.append(name)
        if (folder / "last.pt").exists():
            state = read_checkpoint(folder / "last.pt", torch.device("cpu"))
            last_name = name + "_last"
            target = root / last_name
            target.mkdir(exist_ok=True)
            payload = {key: state[key] for key in ("model", "model_spec", "run_spec", "step")}
            payload["selection"] = {"rule": "last_optimizer_update"}
            atomic_checkpoint(target / "best.pt", payload)
            del state
            names.append(last_name)
    if names and remaining_seconds(journal["deadline"]) > 30:
        command = [sys.executable, "-m", "recsys.cli", "evaluate", "--data", str(data_dir),
                   "--runs", str(root), "--split", "validation", "--names", *names]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=max(1, remaining_seconds(journal["deadline"]) - 10))
        print(result.stdout, flush=True)
        if (root / "evaluation").exists():
            shutil.copytree(root / "evaluation", backup / "evaluation", dirs_exist_ok=True)
        result.check_returncode()
        if remaining_seconds(journal["deadline"]) > 10:
            from .reporting import report
            report(SimpleNamespace(runs=root, output=root / "reports"))
            plot_curves(root, plan)
            shutil.copytree(root / "reports", backup / "reports", dirs_exist_ok=True)
    journal["finished_at"] = time.time()
    write_json(journal_path, journal)
    mirror([journal_path], backup)
    print(f"Extended experiment saved: {backup}. Test has not been evaluated.", flush=True)


def plot_curves(root, plan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name in ["sft"] + [branch["name"] for branch in plan["rl"]]:
        path = root / name / "history.jsonl"
        if not path.exists():
            continue
        entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        points = [entry for entry in entries if "validation_ndcg@10" in entry]
        steps = [entry["step"] for entry in points]
        if name == "sft":
            axes[0].plot(steps, [entry["validation_ndcg@10"] for entry in points], label="SFT continuation")
        else:
            axes[1].plot(steps, [entry["validation_ndcg@10"] for entry in points], label=name)
            axes[2].plot(steps, [entry["validation_reward@10"] for entry in points], label=name)
    for axis, label in zip(axes, ["SFT validation NDCG@10", "RL validation NDCG@10", "RL predicted reward@10"]):
        axis.set(xlabel="Additional optimizer steps", ylabel=label)
        if axis.lines:
            axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "reports/learning_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--source-runs", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--backup", required=True)
    parser.add_argument("--config", default="configs/extended.json")
    parser.add_argument("--allow-cpu", action="store_true", help="Only for tiny integration tests")
    args = parser.parse_args()
    with exclusive_run(args.runs):
        run(args)


if __name__ == "__main__":
    main()
