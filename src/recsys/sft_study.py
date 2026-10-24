"""Controlled SFT data/model comparison, independent of reward checkpoints."""

import argparse
import copy
import csv
import json
import math
import random
import signal
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .budget import exclusive_run, may_train, new_deadline, remaining_seconds
from .data import load_manifest, load_split
from .extended import mirror, update
from .io import sha256, write_json
from .metrics import behavior, paired_bootstrap, per_user_ranking, topk
from .models import Policy
from .scaling import code_identity, evaluate_baselines, evaluation_identity
from .training import atomic_checkpoint, autocast, device, read_checkpoint, seed_all, tensor_batch


def lr_multiplier(step, warmup_steps, max_steps, minimum):
    if warmup_steps and step < warmup_steps:
        return max(1, step + 1) / warmup_steps
    progress = min(1., max(0., (step - warmup_steps) / max(1, max_steps - warmup_steps)))
    return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def measure(model, data, dev, batch_size, popularity=None, deadline=None):
    model.eval()
    losses, slates = [], []
    for start in range(0, len(data["targets"]), batch_size):
        if deadline is not None and remaining_seconds(deadline) < 5:
            raise TimeoutError("Evaluation budget exhausted between batches")
        tokens, target, _ = tensor_batch(data, slice(start, start + batch_size), dev)
        with autocast(dev):
            logits = model(tokens)
            losses.extend(F.cross_entropy(logits, target, reduction="none").float().cpu().tolist())
        slates.append(topk(logits.float().cpu().numpy(), 20))
    slate = np.concatenate(slates)
    per_user = per_user_ranking(slate, data["targets"])
    metrics = {"nll": float(np.mean(losses)), "ranking": {k: float(v.mean()) for k, v in per_user.items()}}
    if popularity is not None:
        metrics["behavior@10"] = behavior(slate[:, :10], data["contexts"], popularity)
    return metrics, per_user, slate


def fit(directory, output, backup, branch, plan, deadline, pilot=False):
    directory, output, backup = Path(directory), Path(output), Path(backup)
    manifest = load_manifest(directory)
    settings = plan["training"]
    train, validation = load_split(directory, "train"), load_split(directory, "validation")
    steps_per_pass = len(train["targets"]) / settings["batch_size"]
    max_steps = plan["pilot_steps"] if pilot else math.ceil(settings["max_passes"] * steps_per_pass)
    interval = max_steps if pilot else max(1, round(settings["eval_passes"] * steps_per_pass))
    warmup_steps = max(0, round(settings["warmup_passes"] * steps_per_pass))
    specification = {"kind": "policy", "kwargs": {"vocab_size": manifest["vocab_size"],
                     "context": manifest["config"]["context"], **branch["model"]}}
    run_spec = {"stage": "sft", "manifest_sha256": sha256(directory / "manifest.json"),
                "settings": settings, "branch": branch, "seed": plan["seed"],
                "max_steps": max_steps, "warmup_steps": warmup_steps, "eval_every": interval,
                "pilot": pilot, "code_sha256": code_identity(), "selection_rule": "full_validation_ndcg"}
    output.mkdir(parents=True, exist_ok=True)
    dev = device()
    seed_all(plan["seed"])
    rng = np.random.default_rng(plan["seed"])
    model = Policy(**specification["kwargs"]).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(
        step, warmup_steps, max_steps, settings["min_lr_ratio"]))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    step, best, anchor, stale, elapsed_before, training_seconds = 0, -float("inf"), -float("inf"), 0, 0., 0.
    reason = None
    if (output / "last.pt").exists():
        state = read_checkpoint(output / "last.pt", dev)
        if state["run_spec"] != run_spec:
            raise ValueError("Cannot resume: code, protocol or SFT settings changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, best, anchor, stale = state["step"], state["best"], state["anchor"], state["stale"]
        elapsed_before, training_seconds = state["elapsed_seconds"], state["training_seconds"]
        reason = state["stop_reason"]
        rng.bit_generator.state = state["sampler_rng"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if dev.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        if reason in {"max_steps", "patience", "stage_time"}:
            print(f"{branch['name']}: already stopped ({reason}), step={step}", flush=True)
            return reason
        reason = None
    elif any(output.iterdir()):
        raise ValueError("Incomplete run without last.pt; preserve it and use a new run directory")
    probe_rng = np.random.default_rng(plan["seed"] + 1)
    probe_indices = np.sort(probe_rng.choice(len(train["targets"]), min(settings["train_probe_examples"], len(train["targets"])), replace=False))
    probe = {key: value[probe_indices] for key, value in train.items()}
    started = last_save = last_progress = time.monotonic()
    last_assessed = None
    interrupted = False
    previous_handler = signal.getsignal(signal.SIGINT)

    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    write_json(output / "run.json", {**run_spec, "model_spec": specification,
        "parameters": sum(p.numel() for p in model.parameters()), "device": str(dev),
        "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else None})

    def elapsed():
        return elapsed_before + time.monotonic() - started

    def save():
        atomic_checkpoint(output / "last.pt", {
            "model": model.state_dict(), "model_spec": specification, "run_spec": run_spec,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "step": step, "best": best, "anchor": anchor, "stale": stale, "stop_reason": reason,
            "elapsed_seconds": elapsed(), "training_seconds": training_seconds,
            "sampler_rng": copy.deepcopy(rng.bit_generator.state), "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if dev.type == "cuda" else [],
        })
        mirror([output / name for name in ["last.pt", "best.pt", "run.json", "history.jsonl"]], backup)

    def assess():
        nonlocal best, anchor, stale, last_assessed
        metrics, _, _ = measure(model, validation, dev, settings["eval_batch_size"])
        probe_metrics, _, _ = measure(model, probe, dev, settings["eval_batch_size"])
        score = metrics["ranking"]["ndcg@10"]
        selected = score > best
        if selected:
            best = score
            atomic_checkpoint(output / "best.pt", {"model": model.state_dict(), "model_spec": specification,
                "run_spec": run_spec, "step": step, "validation_score": score,
                "selection": {"rule": "full_validation_ndcg", "n_users": len(validation["targets"])}})
        if score > anchor + settings["min_delta"]:
            anchor, stale = score, 0
        else:
            stale += 1
        entry = {"step": step, "equivalent_passes": step / steps_per_pass,
                 "elapsed_seconds": elapsed(), "validation_ndcg@10": score,
                 "validation_nll": metrics["nll"], "train_probe_nll": probe_metrics["nll"],
                 "learning_rate": optimizer.param_groups[0]["lr"], "stale_checks": stale, "selected_best": selected}
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
        print(branch["name"], json.dumps(entry), flush=True)
        last_assessed = step
        save()

    signal.signal(signal.SIGINT, request_stop)
    try:
        if not (output / "best.pt").exists():
            assess()
        while step < max_steps and may_train(deadline, plan["evaluation_reserve_minutes"], elapsed(), settings["max_minutes"]):
            if interrupted:
                reason = "interrupted"
                break
            if not pilot and stale >= settings["patience"]:
                reason = "patience"
                break
            model.train()
            indices = rng.integers(0, len(train["targets"]), size=settings["batch_size"])
            tokens, targets, _ = tensor_batch(train, indices, dev)
            optimizer.zero_grad(set_to_none=True)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            update_started = time.monotonic()
            with autocast(dev):
                loss = F.cross_entropy(model(tokens), targets)
            successful = update(loss, model, optimizer, scaler)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            training_seconds += time.monotonic() - update_started
            if not successful:
                continue
            scheduler.step()
            step += 1
            if interrupted:
                reason = "interrupted"
                break
            if step % interval == 0 or step == max_steps:
                assess()
            if time.monotonic() - last_save >= plan["checkpoint_minutes"] * 60:
                save()
                last_save = time.monotonic()
            if time.monotonic() - last_progress >= 30:
                print(f"{branch['name']}: step={step}, passes={step / steps_per_pass:.2f}, minutes={elapsed()/60:.1f}", flush=True)
                last_progress = time.monotonic()
        if reason is None:
            reason = "max_steps" if step >= max_steps else "stage_time" if elapsed() >= settings["max_minutes"] * 60 else "global_budget"
        if last_assessed != step and reason != "interrupted" and remaining_seconds(deadline) > 60:
            assess()
    except KeyboardInterrupt:
        reason = "interrupted"
        raise
    finally:
        try:
            save()
            write_json(output / "timing.json", {"completed_steps": step, "equivalent_passes": step / steps_per_pass,
                "elapsed_seconds": elapsed(), "training_seconds": training_seconds, "stop_reason": reason,
                "updates_per_training_second": step / training_seconds if training_seconds else None,
                "peak_vram_bytes": torch.cuda.max_memory_allocated() if dev.type == "cuda" else 0})
            mirror([output / "timing.json"], backup)
        finally:
            signal.signal(signal.SIGINT, previous_handler)
    if interrupted:
        raise KeyboardInterrupt("Stopped safely after a complete optimizer update")
    return reason


def summarize(data_root, root, plan, deadline=None):
    summary = {"split": "validation", "models": {}, "comparisons": {},
               "note": "Test is not evaluated. User bootstrap does not include training seed uncertainty."}
    expected = evaluation_identity(data_root / "small")
    all_popularity = np.load(data_root / "large/popularity.npy")
    paired = {}
    for variant in ["small", "large"]:
        if deadline is not None and remaining_seconds(deadline) < 30:
            raise TimeoutError("Not enough remaining time for baseline evaluation")
        directory = data_root / variant
        if evaluation_identity(directory) != expected:
            raise ValueError("Cannot compare models with different evaluation/catalog")
        result = evaluate_baselines(directory, root / "evaluation" / variant, reference_popularity=all_popularity)
        for name, metrics in result["models"].items():
            summary["models"][variant + "_" + name] = metrics
    dev = device()
    for branch in plan["branches"]:
        if deadline is not None and remaining_seconds(deadline) < 30:
            raise TimeoutError("Not enough remaining time for policy evaluation")
        path = root / branch["name"] / "best.pt"
        if not path.exists():
            continue
        directory = data_root / branch["dataset"]
        state = read_checkpoint(path, dev)
        if state["run_spec"]["manifest_sha256"] != sha256(directory / "manifest.json"):
            raise ValueError("Policy and training manifest differ")
        model = Policy(**state["model_spec"]["kwargs"]).to(dev)
        model.load_state_dict(state["model"])
        validation = load_split(directory, "validation")
        # One common train-derived popularity reference for comparable behavior metrics.
        metrics, per_user, slate = measure(model, validation, dev, plan["training"]["eval_batch_size"], all_popularity, deadline)
        metrics.update(checkpoint_step=state["step"], checkpoint_sha256=sha256(path))
        summary["models"][branch["name"]] = metrics
        paired[branch["name"]] = per_user
        print(branch["name"], json.dumps(metrics), flush=True)
        (root / "evaluation").mkdir(exist_ok=True)
        np.savez_compressed(root / "evaluation" / (branch["name"] + "_per_user.npz"),
                            users=validation["users"], slate=slate, **per_user)
        del model
    for before, after in [("small_128", "large_128"), ("large_128", "large_256")]:
        if before in paired and after in paired:
            summary["comparisons"][after + "_minus_" + before] = {
                metric: paired_bootstrap(paired[before][metric], paired[after][metric]) for metric in paired[before]}
    summary["evaluation"] = expected
    write_json(root / "evaluation/summary.json", summary)
    rows = []
    for name, metrics in summary["models"].items():
        row = {"model": name, **metrics["ranking"], **metrics["behavior@10"], "step": metrics.get("checkpoint_step")}
        rows.append(row)
    with (root / "evaluation/results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Сравнение объема данных и размера SFT", "",
             "Оценка только на validation; одинаковые даты, каталог и отложенные примеры.", "",
             "| Модель | Recall@10 | NDCG@10 | Coverage@10 | Шаг checkpoint |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['recall@10']:.4f} | {row['ndcg@10']:.4f} | {row['coverage']:.4f} | {row['step'] if row['step'] is not None else '—'} |")
    lines += ["", "Парные интервалы и все метрики: summary.json. Bootstrap не учитывает случайность обучения.",
              "Поведение всех моделей измерено с одной train-популярностью расширенной выборки.",
              "Текущие числа условны на исходной warm-item аудитории; test ранее просматривался и здесь не оценивается.", ""]
    (root / "evaluation/README.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for branch in plan["branches"]:
        path = root / branch["name"] / "history.jsonl"
        if not path.exists():
            continue
        points = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        x = [p["equivalent_passes"] for p in points]
        axes[0].plot(x, [p["validation_ndcg@10"] for p in points], label=branch["name"])
        axes[1].plot(x, [p["validation_nll"] for p in points], label=branch["name"] + " validation")
        axes[1].plot(x, [p["train_probe_nll"] for p in points], linestyle="--", label=branch["name"] + " train")
    for axis, label in zip(axes, ["Validation NDCG@10", "Cross-entropy (eval mode)"]):
        axis.set(xlabel="Equivalent passes (sampling with replacement)", ylabel=label)
        if axis.lines:
            axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "evaluation/learning_curves.png", dpi=150)
    plt.close(fig)
    return summary


def run(data_root, root, backup, plan, pilot=False, allow_cpu=False, session_deadline=None):
    data_root, root, backup = Path(data_root), Path(root), Path(backup)
    if pilot:
        root, backup = root / "pilot", backup / "pilot"
    if root.resolve() == backup.resolve():
        raise ValueError("Local study and backup must be separate")
    if not (root / "study.json").exists() and (backup / "study.json").exists():
        shutil.copytree(backup, root, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("data", "results_scaling_v1.zip"))
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_run(root):
        identity = {"plan": plan, "pilot": pilot, "code_sha256": code_identity(),
                    "data": {name: sha256(data_root / name / "manifest.json") for name in ["small", "large"]}}
        for name in identity["data"]:
            load_manifest(data_root / name)
        if evaluation_identity(data_root / "small") != evaluation_identity(data_root / "large"):
            raise ValueError("Scaling variants have different evaluation protocols")
        if not torch.cuda.is_available() and not allow_cpu:
            raise RuntimeError("Select a GPU runtime; CPU training is limited to explicit tiny tests")
        journal_path = root / "study.json"
        if journal_path.exists():
            journal = json.loads(journal_path.read_text())
            if journal["identity"] != identity:
                raise ValueError("Study code/config/data changed; use a new run directory")
        else:
            deadline = new_deadline(20 if pilot else plan["overall_minutes"])
            if session_deadline is not None:
                deadline = min(deadline, session_deadline)
            journal = {"identity": identity, "deadline": deadline, "completed": []}
            write_json(journal_path, journal)
            mirror([journal_path], backup)
        if remaining_seconds(journal["deadline"]) <= 0:
            print("Study deadline expired. Saved checkpoints remain available; training was not restarted.", flush=True)
            return
        for branch in plan["branches"]:
            if branch["name"] in journal["completed"]:
                continue
            if remaining_seconds(journal["deadline"]) <= plan["evaluation_reserve_minutes"] * 60 + 60:
                break
            stop_reason = fit(data_root / branch["dataset"], root / branch["name"], backup / branch["name"],
                              branch, plan, journal["deadline"], pilot)
            if stop_reason == "global_budget":
                break
            journal["completed"].append(branch["name"])
            write_json(journal_path, journal)
            mirror([journal_path], backup)
        if not pilot and remaining_seconds(journal["deadline"]) > 60:
            try:
                summarize(data_root, root, plan, journal["deadline"])
            except TimeoutError as error:
                print(f"Partial evaluation retained: {error}", flush=True)
            finally:
                if (root / "evaluation").exists():
                    shutil.copytree(root / "evaluation", backup / "evaluation", dirs_exist_ok=True)
        print(f"Saved: {backup}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--backup", required=True)
    parser.add_argument("--config", default="configs/scaling.json")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--session-deadline", type=float)
    args = parser.parse_args()
    run(args.prepared_root, args.runs, args.backup, json.loads(Path(args.config).read_text()),
        args.pilot, args.allow_cpu, args.session_deadline)


if __name__ == "__main__":
    main()
