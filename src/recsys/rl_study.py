"""RL modifications from the selected scaled SFT, reusing an unchanged RM."""

import argparse
import copy
import csv
import json
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .budget import (exclusive_run, may_train, new_deadline, remaining_seconds,
                     rl_study_candidate, rl_study_evaluation_due, rl_study_utility)
from .data import load_manifest, load_split
from .evaluation import candidate_rewards, compatible_reward, evaluate, score_policy
from .extended import mirror, update
from .io import sha256, write_json
from .metrics import paired_bootstrap, per_user_ranking, topk
from .reporting import report
from .rl_objectives import modified_rl_loss
from .rl_frontier import fraction_tag, study_fractions, write_frontier_report
from .rl_schedule import phase_settings, reset_due, retain_snapshot
from .scaling import code_identity
from .training import atomic_checkpoint, autocast, device, freeze, load_model, read_checkpoint, seed_all, tensor_batch


@torch.no_grad()
def measure(policy, reward, data, dev, batch_size):
    policy.eval()
    ndcg, rewards, repeats = [], [], []
    for start in range(0, len(data["targets"]), batch_size):
        tokens, targets, _ = tensor_batch(data, slice(start, start + batch_size), dev)
        with autocast(dev):
            slate_np = topk(policy(tokens).float().cpu().numpy(), 10)
            slate = torch.as_tensor(slate_np, device=dev)
            predicted = candidate_rewards(reward, tokens, slate).mean(dim=1)
        ndcg.extend(per_user_ranking(slate_np, targets.cpu().numpy(), (10,))["ndcg@10"].tolist())
        rewards.extend(predicted.float().cpu().tolist())
        repeats.extend(slate[:, :, None].eq(tokens[:, None, :]).any(dim=2).float().mean(dim=1).cpu().tolist())
    return {"ndcg@10": float(np.mean(ndcg)), "rm_reward@10": float(np.mean(rewards)),
            "seen_fraction": float(np.mean(repeats))}


def initialize(args, plan):
    data, root, backup = Path(args.data), Path(args.runs), Path(args.backup)
    source_sft, source_rm = Path(args.source_sft), Path(args.source_rm)
    if root.resolve() == backup.resolve() or root.resolve() in {data.resolve(), source_sft.parent.parent.resolve(), source_rm.parent.parent.resolve()}:
        raise ValueError("RL study must use separate output and backup directories")
    manifest = load_manifest(data)
    identity = {"plan": plan, "code_sha256": code_identity(), "manifest_sha256": sha256(data / "manifest.json"),
                "sft_sha256": sha256(source_sft), "rm_sha256": sha256(source_rm),
                "reward_source_manifest_sha256": sha256(Path(args.reward_source_data) / "manifest.json")}
    for key, actual in [("expected_sft_sha256", identity["sft_sha256"]), ("expected_manifest_sha256", identity["manifest_sha256"]),
                        ("expected_rm_sha256", identity["rm_sha256"])]:
        if plan.get(key) and plan[key] != actual:
            raise ValueError(f"Selected scaling model/data differ: {key}")
    if plan.get("comparison_archive"):
        from .rl_ablation import validate_reference
        validate_reference(Path(args.config).resolve().parents[1] / plan["comparison_archive"], plan, identity)
    if not (root / "experiment.json").exists() and (backup / "experiment.json").exists():
        shutil.copytree(backup, root, dirs_exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    journal_path = root / "experiment.json"
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        if journal["identity"] != identity:
            raise ValueError("Existing RL experiment has different code, inputs or settings")
    else:
        policy, state = load_model(source_sft, torch.device("cpu"))
        if state["model_spec"]["kind"] != "policy" or state["run_spec"]["stage"] != "sft" or state["run_spec"]["manifest_sha256"] != identity["manifest_sha256"]:
            raise ValueError("Expected selected SFT trained on the expanded protocol")
        reward, _, transfer = compatible_reward(source_rm, data, torch.device("cpu"), args.reward_source_data)
        del policy, reward
        journal = {"identity": identity, "started_at": time.time(), "deadline": new_deadline(plan["overall_minutes"]),
                   "completed": [], "pilot_completed": [], "reward_transfer": transfer}
        write_json(journal_path, journal)
    for name, source in [("sft", source_sft), ("rm", source_rm)]:
        target = root / name / "best.pt"
        target.parent.mkdir(exist_ok=True)
        if target.exists() and sha256(target) != sha256(source):
            raise ValueError("A frozen study checkpoint changed")
        if not target.exists():
            shutil.copy2(source, target)
        mirror([target], backup / name)
    write_json(root / "config.json", plan)
    mirror([journal_path, root / "config.json"], backup)
    return journal


def fit(args, settings, plan, journal, pilot=False):
    root, backup, data_dir = Path(args.runs), Path(args.backup), Path(args.data)
    name = settings["name"]
    output = root / ("pilot" if pilot else "branches") / name
    destination = backup / ("pilot" if pilot else "branches") / name
    output.mkdir(parents=True, exist_ok=True)
    dev = device()
    training_seed = settings.get("seed", plan["seed"])
    seed_all(training_seed)
    rng = np.random.default_rng(training_seed)
    policy, source = load_model(root / "sft/best.pt", dev)
    reference, _ = load_model(root / "sft/best.pt", dev)
    freeze(reference)
    reward, _, _ = compatible_reward(root / "rm/best.pt", data_dir, dev, args.reward_source_data)
    run_spec = {"stage": "rl", "manifest_sha256": journal["identity"]["manifest_sha256"],
        "settings": settings, "seed": training_seed, "code_sha256": journal["identity"]["code_sha256"],
        "dependencies": {"init_sha256": journal["identity"]["sft_sha256"], "reward_sha256": journal["identity"]["rm_sha256"]},
        "selection_rule": "validation_rm_minus_repeat_penalty_with_ndcg_floor", "relevance_fraction": plan["relevance_fraction"]}
    if settings.get("reinforce_weight", 1.) == 0:
        run_spec["stage"] = "sft_control"
    optimizer = torch.optim.AdamW(policy.parameters(), lr=settings["learning_rate"], weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    train, validation = load_split(data_dir, "train"), load_split(data_dir, "validation")
    reference_metrics = json.loads((root / "reference.json").read_text())
    step, baseline, elapsed_before, best = 0, 0., 0., dict(reference_metrics)
    optimization_seconds = 0.
    training_phase, optimizer_resets, skipped_updates = 1, 0, 0
    fractions = study_fractions(plan)
    best_by_fraction = {fraction_tag(f): dict(reference_metrics) for f in fractions}
    if (output / "last.pt").exists():
        saved = read_checkpoint(output / "last.pt", dev)
        if saved["run_spec"] != run_spec:
            raise ValueError("Resume settings differ")
        policy.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        step, baseline, elapsed_before, best = saved["step"], saved["baseline"], saved["elapsed_seconds"], saved["best_metrics"]
        optimization_seconds = saved.get("optimization_seconds", 0.)
        training_phase = saved.get("training_phase", 1)
        optimizer_resets = saved.get("optimizer_resets", 0)
        skipped_updates = saved.get("skipped_updates", 0)
        best_by_fraction = saved.get("best_metrics_by_fraction", best_by_fraction)
        if "selection_fractions" in plan:
            # The compact checkpoint is authoritative if Drive mirroring stopped between files.
            for fraction in fractions:
                path = output / ("best_" + fraction_tag(fraction) + ".pt")
                if path.exists():
                    best_by_fraction[fraction_tag(fraction)] = read_checkpoint(path, torch.device("cpu"))["selection"]["metrics"]
            if (output / "best.pt").exists():
                best = read_checkpoint(output / "best.pt", torch.device("cpu"))["selection"]["metrics"]
        rng.bit_generator.state = saved["sampler_rng"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if dev.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])
    elif any(output.iterdir()):
        raise ValueError("Incomplete RL folder without last.pt; preserve it and use a new experiment directory")
    started = last_save = last_progress = time.monotonic()
    interrupted = False
    previous_handler = signal.getsignal(signal.SIGINT)
    last_assessed = None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    write_json(output / "run.json", {**run_spec, "model_spec": source["model_spec"],
               "reference_metrics": reference_metrics, "reward_transfer": journal["reward_transfer"]})

    def elapsed():
        return elapsed_before + time.monotonic() - started

    def compact(metrics, rule, fraction=None):
        return {"model": policy.state_dict(), "model_spec": source["model_spec"], "run_spec": run_spec, "step": step,
                "training_phase": training_phase,
                "selection": {"rule": rule, "metrics": metrics, "repeat_penalty": settings["repeat_penalty"],
                              "reference_metrics": reference_metrics,
                              "relevance_fraction": plan["relevance_fraction"] if fraction is None else fraction}}

    def save():
        atomic_checkpoint(output / "last.pt", {
            **compact({}, "last_optimizer_update"), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "baseline": baseline, "elapsed_seconds": elapsed(), "best_metrics": best,
            "optimization_seconds": optimization_seconds,
            "training_phase": training_phase, "optimizer_resets": optimizer_resets, "skipped_updates": skipped_updates,
            "best_metrics_by_fraction": best_by_fraction,
            "sampler_rng": copy.deepcopy(rng.bit_generator.state), "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if dev.type == "cuda" else [],
        })
        mirror([output / f for f in ["last.pt", "run.json", "history.jsonl"]], destination)

    def assess(stats=None, initial=False):
        nonlocal best, last_assessed
        metrics = dict(reference_metrics) if initial else measure(policy, reward, validation, dev, 64)
        feasible = metrics["ndcg@10"] >= plan["relevance_fraction"] * reference_metrics["ndcg@10"]
        selected = not (output / "best.pt").exists() or rl_study_candidate(
            metrics, best, reference_metrics, plan["relevance_fraction"], settings["repeat_penalty"])
        snapshot = output / f"step_{step:07d}.pt"
        keep_snapshot = retain_snapshot(settings, step)
        if keep_snapshot:
            atomic_checkpoint(snapshot, compact(metrics, "intermediate_snapshot"))
        if selected:
            best = metrics
            atomic_checkpoint(output / "best.pt", compact(metrics, run_spec["selection_rule"]))
        changed = []
        if "selection_fractions" in plan:
            for fraction in fractions:
                key = fraction_tag(fraction)
                path = output / ("best_" + key + ".pt")
                if not path.exists() or rl_study_candidate(metrics, best_by_fraction[key], reference_metrics,
                                                          fraction, settings["repeat_penalty"]):
                    best_by_fraction[key] = dict(metrics)
                    atomic_checkpoint(path, compact(metrics, run_spec["selection_rule"], fraction))
                    changed.append(path)
        entry = {"step": step, "elapsed_seconds": elapsed(), "validation_ndcg@10": metrics["ndcg@10"],
                 "validation_reward@10": metrics["rm_reward@10"], "validation_seen_fraction": metrics["seen_fraction"],
                 "validation_utility": rl_study_utility(metrics, settings["repeat_penalty"]),
                 "feasible": feasible, "selected_best": selected, **(stats or {})}
        if "selection_fractions" in plan:
            entry["selected_floors"] = [p.stem.removeprefix("best_") for p in changed]
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
        print(name, json.dumps(entry, allow_nan=False), flush=True)
        mirror(([snapshot] if keep_snapshot else []) + [output / "history.jsonl", *changed] + ([output / "best.pt"] if selected else []), destination)
        last_assessed = step
        save()

    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        if not (output / "best.pt").exists():
            assess(initial=True)
        while step < settings["max_steps"] and may_train(journal["deadline"], plan["evaluation_reserve_minutes"], elapsed(), settings["max_minutes"]):
            if interrupted:
                break
            if reset_due(settings, step, training_phase):
                optimizer = torch.optim.AdamW(policy.parameters(), lr=settings["learning_rate"], weight_decay=.01)
                scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
                baseline = 0.
                training_phase = 2
                optimizer_resets += 1
                print(f"{name}: phase 2 at update {step}; optimizer/scaler/baseline reset", flush=True)
            effective_settings, phase = phase_settings(settings, step)
            if phase != training_phase:
                raise ValueError("Saved optimizer phase is inconsistent with the schedule")
            policy.eval()  # dropout-free sampling and log probabilities, gradients remain enabled
            update_started = time.monotonic()
            indices = rng.integers(0, len(train["targets"]), size=settings["batch_size"])
            tokens, targets, _ = tensor_batch(train, indices, dev)
            optimizer.zero_grad(set_to_none=True)
            with autocast(dev):
                loss, stats = modified_rl_loss(policy, reference, reward, tokens, targets, effective_settings, baseline)
                if settings.get("training_schedule"):
                    stats["training_phase"] = training_phase
            if not update(loss, policy, optimizer, scaler):
                skipped_updates += 1
                continue
            if pilot and dev.type == "cuda":
                torch.cuda.synchronize()
            optimization_seconds += time.monotonic() - update_started
            baseline = .95 * baseline + .05 * stats["objective_return"]
            step += 1
            if interrupted:
                break
            if rl_study_evaluation_due(step, settings["eval_every"]) or step == settings["max_steps"]:
                assess({**stats, "loss": float(loss.detach())})
            if time.monotonic() - last_save >= plan["checkpoint_minutes"] * 60:
                save()
                last_save = time.monotonic()
            if time.monotonic() - last_progress >= 30:
                print(f"{name}: step={step}, minutes={elapsed()/60:.1f}, remaining={remaining_seconds(journal['deadline'])/60:.1f} min", flush=True)
                last_progress = time.monotonic()
        if not interrupted and last_assessed != step and remaining_seconds(journal["deadline"]) > 60:
            assess()
    finally:
        try:
            save()
            reason = "interrupted" if interrupted else "max_steps" if step >= settings["max_steps"] else "stage_time" if elapsed() >= settings["max_minutes"] * 60 else "global_budget"
            write_json(output / "timing.json", {"completed_steps": step, "elapsed_seconds": elapsed(), "stop_reason": reason,
                "optimization_seconds": optimization_seconds,
                "optimization_seconds_per_update": optimization_seconds / step if step else None,
                "training_phase": training_phase, "optimizer_resets": optimizer_resets, "skipped_updates": skipped_updates,
                "best_metrics": best, "peak_vram_bytes": torch.cuda.max_memory_allocated() if dev.type == "cuda" else 0})
            mirror([output / "timing.json"], destination)
        finally:
            signal.signal(signal.SIGINT, previous_handler)
    if interrupted:
        raise KeyboardInterrupt("RL saved after a complete optimizer update")
    return reason


def evaluate_all(args, plan, full=True):
    root, data_dir = Path(args.runs), Path(args.data)
    names = ["sft", "rerank"]
    selections, last_aliases = {}, {}
    if full:
        for branch in plan["branches"]:
            folder = root / "branches" / branch["name"]
            options = [(branch["name"], "best.pt", plan["relevance_fraction"])]
            if "selection_fractions" in plan:
                options.extend((branch["name"] + "_" + fraction_tag(f), "best_" + fraction_tag(f) + ".pt", f)
                               for f in study_fractions(plan) if f != plan["relevance_fraction"])
            options.append((branch["name"] + "_last", "last.pt", None))
            canonical_by_step = {}
            for label, filename, fraction in options:
                if not (folder / filename).exists():
                    continue
                state = read_checkpoint(folder / filename, torch.device("cpu"))
                step = state["step"]
                if "selection_fractions" in plan and step in canonical_by_step:
                    canonical = canonical_by_step[step]
                else:
                    canonical = label
                    target = root / canonical
                    target.mkdir(exist_ok=True)
                    if filename != "last.pt":
                        shutil.copy2(folder / filename, target / "best.pt")
                    else:
                        compact = {k: state[k] for k in ["model", "model_spec", "run_spec", "step"]}
                        if "training_phase" in state:
                            compact["training_phase"] = state["training_phase"]
                        compact["selection"] = {"rule": "last_optimizer_update"}
                        atomic_checkpoint(target / "best.pt", compact)
                    names.append(canonical)
                    canonical_by_step[step] = canonical
                if fraction is not None:
                    selections.setdefault(branch["name"], {})[fraction_tag(fraction)] = {
                        "model": canonical, "checkpoint_step": step, "fraction": fraction}
                else:
                    last_aliases[branch["name"]] = canonical
                del state
    rollout_names = None
    if "selection_fractions" in plan:
        rollout_names = ["sft"] + [b["name"] for b in plan["branches"]]
    evaluate(SimpleNamespace(data=data_dir, runs=root, split="validation", names=names,
        reward_model="rm", reward_source_data=args.reward_source_data, rerank_lambda=1., quiet=True, progress=True,
        rollout_names=rollout_names))
    summary_path = root / "evaluation/validation/summary.json"
    summary = json.loads(summary_path.read_text())
    dev = device()
    reference, _ = load_model(root / "sft/best.pt", dev)
    freeze(reference)
    reward, _, _ = compatible_reward(root / "rm/best.pt", data_dir, dev, args.reward_source_data)
    validation = load_split(data_dir, "validation")
    popularity = np.load(data_dir / "popularity.npy")
    with np.load(root / "evaluation/validation/sft_per_user.npz", allow_pickle=False) as z:
        sft_values = {k: z[k] for k in ["recall@10", "ndcg@10", "recall@20", "ndcg@20", "rm_reward@10"]}
    for strength in plan["rerank_lambdas"]:
        if strength == 1:
            continue
        label = f"rerank_{strength:g}"
        metrics, values, _, slate = score_policy(reference, reference, reward, validation, popularity, dev, 64, strength)
        metrics.update({"lambda": strength, "checkpoint_sha256": sha256(root / "sft/best.pt")})
        summary["models"][label] = metrics
        summary["paired_vs_sft"][label] = {k: paired_bootstrap(sft_values[k], v) for k, v in values.items()}
        np.savez_compressed(root / f"evaluation/validation/{label}_per_user.npz", users=validation["users"],
                            targets=validation["targets"], slate=slate, **values)
    summary["paired_vs_kl03"] = {}
    control_path = root / "evaluation/validation/rl_kl03_per_user.npz"
    if full and control_path.exists():
        with np.load(control_path, allow_pickle=False) as z:
            control_users = z["users"]
            control = {k: z[k] for k in sft_values}
        for name in ["rl_repeat", "rl_anchor", "rl_loo", "rl_h5"]:
            path = root / f"evaluation/validation/{name}_per_user.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(control_users, z["users"])
                summary["paired_vs_kl03"][name] = {k: paired_bootstrap(control[k], z[k]) for k in control}
    summary["rl_study_note"] = "Repeat-penalty branches select RM reward minus penalty*seen_fraction under the NDCG floor; other branches select raw RM reward. LOO is REINFORCE with a leave-one-out baseline, not PPO/GRPO."
    if "selection_fractions" in plan:
        summary["selection_by_constraint"] = selections
        summary["last_checkpoint_aliases"] = last_aliases
    write_json(summary_path, summary)
    report(SimpleNamespace(runs=root, output=root / "reports", relevance_fraction=plan["relevance_fraction"],
        candidate_names=[b["name"] for b in plan["branches"]] if "selection_fractions" in plan else None))
    if "selection_fractions" in plan and full:
        write_frontier_report(root, summary, plan, selections)
        if plan.get("comparison_archive"):
            from .rl_ablation import compare_exports
            compare_exports(root, Path(args.config).resolve().parents[1] / plan["comparison_archive"], plan)
    with (root / "reports/README.md").open("a") as stream:
        stream.write("\n## Модификации RL\n\nВетки со штрафом повторов выбирают checkpoint по `RM reward@10 − penalty × seen_fraction` при ограничении NDCG; остальные — по RM reward при том же ограничении. LOO — REINFORCE с leave-one-out baseline, без PPO clipping. RM перенесена без изменения весов из исходного эксперимента; ее исходный manifest сохранен. Улучшение ее прогноза не доказывает улучшения отклика пользователя.\n")
    rows = [{"model": name, "step": m.get("checkpoint_step"), "ndcg@10": m["ranking"]["ndcg@10"],
             "recall@10": m["ranking"]["recall@10"], "rm_reward@10": m.get("rm_reward@10"),
             "seen_fraction": m["behavior@10"]["seen_fraction"], "coverage": m["behavior@10"]["coverage"]}
            for name, m in summary["models"].items()]
    with (root / "reports/results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if full:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for branch in plan["branches"]:
            path = root / "branches" / branch["name"] / "history.jsonl"
            if not path.exists():
                continue
            points = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for axis, key in zip(axes.flat, ["validation_ndcg@10", "validation_reward@10", "validation_seen_fraction", "validation_utility"]):
                axis.plot([p["step"] for p in points], [p[key] for p in points], marker=".", label=branch["name"])
        axes[0, 0].axhline(plan["relevance_fraction"] * summary["models"]["sft"]["ranking"]["ndcg@10"], color="black", linestyle="--", label="NDCG floor")
        for axis, label in zip(axes.flat, ["Validation NDCG@10", "Predicted RM reward@10", "Seen fraction@10", "Branch utility (penalty differs)"]):
            axis.set(xlabel="RL updates", ylabel=label)
            if axis.lines:
                axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root / "reports/rl_learning_curves.png", dpi=150)
        plt.close(fig)
    if not full:
        metrics = summary["models"]["sft"]
        write_json(root / "reference.json", {"ndcg@10": metrics["ranking"]["ndcg@10"],
                   "rm_reward@10": metrics["rm_reward@10"], "seen_fraction": metrics["behavior@10"]["seen_fraction"]})
        shutil.copytree(root / "evaluation", root / "initial_evaluation", dirs_exist_ok=True)
    shutil.copytree(root / "evaluation", Path(args.backup) / "evaluation", dirs_exist_ok=True)
    shutil.copytree(root / "reports", Path(args.backup) / "reports", dirs_exist_ok=True)
    mirror([root / "reference.json"], args.backup)
    print("Model | step | NDCG@10 | RM reward@10 | seen_fraction", flush=True)
    for name, metrics in summary["models"].items():
        print(name, metrics.get("checkpoint_step"), metrics["ranking"]["ndcg@10"],
              metrics.get("rm_reward@10"), metrics["behavior@10"]["seen_fraction"], flush=True)
    print("RM quality on observed targets:", json.dumps({k: {m: v[m] for m in ["mae", "rmse", "spearman"]}
        for k, v in summary["reward_model_evaluation"].items()}), flush=True)


def bounded_evaluation(args, plan, journal, full):
    seconds = remaining_seconds(journal["deadline"]) - 10
    if seconds < 30:
        print("Not enough remaining time for evaluation; checkpoints are retained.", flush=True)
        return
    command = [sys.executable, "-m", "recsys.rl_study", "--stage", "evaluate" if full else "baseline",
               "--data", args.data, "--runs", args.runs, "--backup", args.backup, "--source-sft", args.source_sft,
               "--source-rm", args.source_rm, "--reward-source-data", args.reward_source_data, "--config", args.config]
    if args.allow_cpu:
        command.append("--allow-cpu")
    command.append("--evaluation-child")
    with subprocess.Popen(list(map(str, command))) as process:
        try:
            code = process.wait(timeout=seconds)
            if code:
                raise subprocess.CalledProcessError(code, command)
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=60)
            raise
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=60)
            print("Evaluation time limit reached; partial results retained.", flush=True)
        finally:
            if (Path(args.runs) / "evaluation").exists():
                shutil.copytree(Path(args.runs) / "evaluation", Path(args.backup) / "evaluation", dirs_exist_ok=True)


def run(args):
    plan = json.loads(Path(args.config).read_text())
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("Use a GPU runtime")
    # Evaluation is a bounded child of the lock-holding controller.
    if args.stage in {"evaluate", "baseline"}:
        if args.evaluation_child:
            evaluate_all(args, plan, args.stage == "evaluate")
        else:
            with exclusive_run(args.runs):
                journal = initialize(args, plan)
                bounded_evaluation(args, plan, journal, args.stage == "evaluate")
        return
    with exclusive_run(args.runs):
        journal = initialize(args, plan)
        if remaining_seconds(journal["deadline"]) <= 0:
            print("RL experiment deadline expired; saved results remain available.", flush=True)
            return
        root = Path(args.runs)
        if not (root / "reference.json").exists():
            bounded_evaluation(args, plan, journal, False)
        if args.stage == "prepare":
            return
        if not (root / "reference.json").exists():
            raise RuntimeError("Reference evaluation has not completed")
        if args.stage == "pilot":
            branches = plan.get("pilot_branches", [{"name": "pilot_h1"}, {"name": "pilot_loo", "baseline_mode": "loo", "group_size": 4},
                        {"name": "pilot_h5", "horizon": 5, "repeat_penalty": .2, "ce_weight": .1}])
        else:
            branches = plan["branches"]
        completed_key = "pilot_completed" if args.stage == "pilot" else "completed"
        for branch in branches:
            if branch["name"] in journal[completed_key]:
                continue
            if remaining_seconds(journal["deadline"]) <= (plan["evaluation_reserve_minutes"] + 1) * 60:
                break
            settings = {**plan["training"], **branch}
            if args.stage == "pilot":
                settings.update(max_steps=plan.get("pilot_steps", 3), max_minutes=2)
                if settings.get("training_schedule"):
                    settings["switch_step"] = plan.get("pilot_switch_step", max(1, settings["max_steps"] // 2))
            reason = fit(args, settings, plan, journal, args.stage == "pilot")
            if reason == "global_budget":
                break
            journal[completed_key].append(branch["name"])
            write_json(root / "experiment.json", journal)
            mirror([root / "experiment.json"], args.backup)
        if args.stage == "train":
            bounded_evaluation(args, plan, journal, True)
        print("Results:", args.backup, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "pilot", "train", "evaluate", "baseline"], required=True)
    for name in ["data", "runs", "backup", "source-sft", "source-rm", "reward-source-data"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--config", default="configs/rl_study.json")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--evaluation-child", action="store_true", help=argparse.SUPPRESS)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
