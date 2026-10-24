"""Bounded single-GPU training with atomic, resumable checkpoints."""

import copy
import json
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_manifest, load_split
from .io import sha256, write_json
from .metrics import per_user_ranking, topk
from .models import Policy, RewardModel, rl_loss


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(dev):
    return torch.autocast("cuda", dtype=torch.float16) if dev.type == "cuda" else nullcontext()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(spec):
    if spec["kind"] == "policy":
        return Policy(**spec["kwargs"])
    if spec["kind"] == "reward":
        return RewardModel(**spec["kwargs"])
    raise ValueError(f"Unknown model kind: {spec['kind']}")


def read_checkpoint(path, dev):
    # These files contain our RNG/optimizer state: only load project-created checkpoints.
    return torch.load(path, map_location=dev, weights_only=False)


def load_model(path, dev):
    checkpoint = read_checkpoint(path, dev)
    model = create_model(checkpoint["model_spec"]).to(dev)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def freeze(model):
    model.eval()
    model.requires_grad_(False)
    return model


def tensor_batch(data, indices, dev):
    return (
        torch.as_tensor(data["contexts"][indices], dtype=torch.long, device=dev),
        torch.as_tensor(data["targets"][indices], dtype=torch.long, device=dev),
        torch.as_tensor(data["rewards"][indices], dtype=torch.float32, device=dev),
    )


@torch.no_grad()
def validate(model, data, stage, dev, batch_size=64):
    model.eval()
    values = []
    for start in range(0, len(data["targets"]), batch_size):
        tokens, target, reward = tensor_batch(data, slice(start, start + batch_size), dev)
        with autocast(dev):
            if stage == "rm":
                predicted = model(tokens, target).float()
                valid = torch.isfinite(reward)
                values.extend(((predicted[valid] - reward[valid]) ** 2).cpu().tolist())
            else:
                scores = model(tokens).float().cpu().numpy()
                values.extend(per_user_ranking(topk(scores, 10), target.cpu().numpy(), (10,))["ndcg@10"].tolist())
    if not values:
        raise ValueError("No valid validation examples/rewards")
    value = float(np.mean(values))
    return -value if stage == "rm" else value


def atomic_checkpoint(path, state):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(args):
    data_dir, run_root = Path(args.data), Path(args.runs)
    manifest = load_manifest(data_dir)
    config = manifest["config"]
    stage = args.stage
    settings = dict(config[stage])
    if args.max_steps is not None:
        settings["max_steps"] = args.max_steps
    if args.max_minutes is not None:
        settings["max_minutes"] = args.max_minutes
    if settings["max_steps"] < 1 or settings["max_minutes"] <= 0:
        raise ValueError("Positive step/time limits are required")
    if args.beta < 0 or args.repeat_penalty < 0:
        raise ValueError("KL and repeat penalties must be nonnegative")
    name = args.name or stage
    output = run_root / name
    output.mkdir(parents=True, exist_ok=True)
    last_path = output / "last.pt"
    if args.resume and not last_path.exists():
        raise ValueError(f"Cannot resume: {last_path} does not exist")
    if not args.resume and any(output.iterdir()):
        raise ValueError(f"Run {name} already exists; use --resume or a new --name")
    dev = device()
    seed_all(config["seed"])
    rng = np.random.default_rng(config["seed"])
    train_data = load_split(data_dir, "train")
    validation = load_split(data_dir, "validation", config["validation_train_limit"])
    if stage == "rm":
        valid = np.isfinite(train_data["rewards"])
        train_data = {key: value[valid] for key, value in train_data.items()}
    if not len(train_data["targets"]):
        raise ValueError("No training examples for this stage")
    spec = {"kind": "reward", "kwargs": {"vocab_size": manifest["vocab_size"], "hidden": 64}}
    reference = reward_model = None
    dependencies = {}
    if stage != "rm":
        spec = {"kind": "policy", "kwargs": {
            "vocab_size": manifest["vocab_size"], "context": config["context"], **config["model"]}}
    if stage == "rl":
        init_path = run_root / args.init / "best.pt"
        rm_path = run_root / args.reward_model / "best.pt"
        reference, ref_checkpoint = load_model(init_path, dev)
        reward_model, rm_checkpoint = load_model(rm_path, dev)
        expected_manifest = sha256(data_dir / "manifest.json")
        if ref_checkpoint["run_spec"]["manifest_sha256"] != expected_manifest or rm_checkpoint["run_spec"]["manifest_sha256"] != expected_manifest:
            raise ValueError("SFT/RM checkpoints belong to another data protocol")
        if ref_checkpoint["run_spec"]["stage"] != "sft" or rm_checkpoint["model_spec"]["kind"] != "reward":
            raise ValueError("RL requires an SFT reference and a reward model")
        freeze(reference)
        freeze(reward_model)
        spec = ref_checkpoint["model_spec"]
        dependencies = {"init_sha256": sha256(init_path), "reward_sha256": sha256(rm_path)}
    model = create_model(spec).to(dev)
    if reference is not None:
        model.load_state_dict(reference.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    run_spec = {
        "stage": stage, "manifest_sha256": sha256(data_dir / "manifest.json"),
        "settings": settings, "seed": config["seed"], "dependencies": dependencies,
        "beta": args.beta if stage == "rl" else None,
        "repeat_penalty": args.repeat_penalty if stage == "rl" else None,
        "horizon": args.horizon if stage == "rl" else None,
    }
    step, best, elapsed_before, baseline = 0, -float("inf"), 0., 0.
    if args.resume:
        saved = read_checkpoint(last_path, dev)
        if saved["run_spec"] != run_spec or saved["model_spec"] != spec:
            raise ValueError("Resume configuration/data/dependencies differ from the saved run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        step, best = saved["step"], saved["best"]
        elapsed_before, baseline = saved["elapsed_seconds"], saved["baseline"]
        rng.bit_generator.state = saved["sampler_rng"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if dev.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    write_json(output / "run.json", {**run_spec, "model_spec": spec, "device": str(dev),
                                    "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else None})
    started = time.monotonic()
    checkpoint_at = started
    history_path = output / "history.jsonl"

    def backup():
        if args.backup:
            destination = Path(args.backup) / name
            destination.mkdir(parents=True, exist_ok=True)
            for source in output.iterdir():
                if source.is_file() and not source.name.endswith(".tmp"):
                    temporary = destination / (source.name + ".tmp")
                    shutil.copy2(source, temporary)
                    temporary.replace(destination / source.name)

    def save():
        state = {
            "model_spec": spec, "run_spec": run_spec, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "step": step, "best": best, "baseline": baseline,
            "elapsed_seconds": elapsed_before + time.monotonic() - started,
            "sampler_rng": copy.deepcopy(rng.bit_generator.state),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if dev.type == "cuda" else [],
        }
        atomic_checkpoint(last_path, state)
        backup()

    try:
        while step < settings["max_steps"]:
            elapsed = elapsed_before + time.monotonic() - started
            if elapsed >= settings["max_minutes"] * 60:
                break
            # RL must use the same dropout-free policy for sampling/log-probability.
            model.train(stage != "rl")
            indices = rng.integers(0, len(train_data["targets"]), size=settings["batch_size"])
            tokens, targets, rewards = tensor_batch(train_data, indices, dev)
            optimizer.zero_grad(set_to_none=True)
            stats = {}
            with autocast(dev):
                if stage == "sft":
                    loss = F.cross_entropy(model(tokens), targets)
                elif stage == "rm":
                    loss = F.mse_loss(model(tokens, targets).float(), rewards)
                else:
                    loss, stats = rl_loss(model, reference, reward_model, tokens, args.beta,
                                          args.repeat_penalty, args.horizon, settings["gamma"], baseline)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; last checkpoint retained")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            finite_gradients = torch.stack([torch.isfinite(g).all() for g in gradients]).all()
            if not finite_gradients:
                if not scaler.is_enabled():
                    raise FloatingPointError("Non-finite gradients without AMP; last checkpoint retained")
                # unscale_ recorded overflow. Let GradScaler skip the update and
                # reduce its scale; do not clip or count this as an optimizer step.
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                print(f"AMP overflow: skipped update, scale {previous_scale:g} -> {scaler.get_scale():g}", flush=True)
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite gradients; last checkpoint retained")
            scaler.step(optimizer)
            scaler.update()
            if stage == "rl":
                baseline = .95 * baseline + .05 * stats["objective_return"]
            step += 1
            entry = {"step": step, "loss": float(loss.detach()), "elapsed_seconds": elapsed_before + time.monotonic() - started, **stats}
            if step % settings["eval_every"] == 0 or step == 1 or step == settings["max_steps"]:
                score = validate(model, validation, stage, dev, config["eval_batch_size"])
                entry["validation_score"] = score
                if score > best:
                    best = score
                    # Compact inference snapshot; latest full training state is separate.
                    atomic_checkpoint(output / "best.pt", {
                        "model_spec": spec, "run_spec": run_spec, "model": model.state_dict(),
                        "step": step, "validation_score": score,
                    })
                print(json.dumps(entry), flush=True)
            if step % 50 == 0 or "validation_score" in entry:
                with history_path.open("a") as stream:
                    stream.write(json.dumps(entry, allow_nan=False) + "\n")
            if step == 1 or time.monotonic() - checkpoint_at >= config["checkpoint_minutes"] * 60:
                save()
                checkpoint_at = time.monotonic()
    finally:
        save()
        write_json(output / "timing.json", {
            "completed_steps": step, "elapsed_seconds": elapsed_before + time.monotonic() - started,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if dev.type == "cuda" else 0,
            "budget_exhausted": elapsed_before + time.monotonic() - started >= settings["max_minutes"] * 60,
        })
        backup()
    print(f"Saved {output}; steps={step}, device={dev}", flush=True)
