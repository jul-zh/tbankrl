"""Resumable evaluation of frozen validation choices; never trains or selects."""

import argparse
import csv
import json
import pickle
import platform
import re
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

from .io import sha256, write_json


def read_selection(path):
    selection = json.loads(Path(path).read_text())
    if selection.get("test_driven_reselection_allowed") is not False:
        raise ValueError("Final evaluation requires a frozen selection without test reselection")
    models = selection["models"]
    names = [m["name"] for m in models]
    if len(set(names)) != len(names) or names.count("sft") != 1:
        raise ValueError("Expected unique model names and exactly one SFT")
    for m in models:
        path = Path(m["source_relative_to_drive_base"])
        if not re.fullmatch(r"[a-zA-Z0-9_]+", m["name"]) or path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid frozen model path or name")
    primary = [m for m in models if m["family"] == selection["primary_family"]]
    if sorted(m["seed"] for m in primary) != sorted(selection["seeds"]):
        raise ValueError("Primary family must include every frozen seed exactly once")
    return selection


def resolve_sources(base, selection, search_roots=()):
    """Recover only the same stored checkpoint or a hash-locked compact copy."""
    base = Path(base)
    sources, missing = {}, []
    for entry in selection["models"]:
        relative = Path(entry["source_relative_to_drive_base"])
        original = base / relative
        expected = entry.get("source_file_sha256")
        options = [(original, expected, "original")]
        study = relative.parts[0]
        study_roots = [base / study]
        for root in search_roots:
            root = Path(root)
            options.append((root / ("artifacts_" + study) / Path(*relative.parts[1:]), expected, "same_checkpoint_local_copy"))
            options.append((root / relative, expected, "same_checkpoint_local_copy"))
            study_roots.extend([root / ("artifacts_" + study), root / study])
        alias = entry.get("validation_evaluation_alias")
        compact_hash = entry.get("validation_evaluated_file_sha256")
        if alias and compact_hash:
            branch = relative.parent.name
            for folder in study_roots:
                options.append((folder / alias / "best.pt", compact_hash, "hash_locked_validation_copy"))
                if alias.startswith(branch + "_floor_"):
                    filename = "best_" + alias[len(branch) + 1:] + ".pt"
                    options.append((folder / "branches" / branch / filename, compact_hash, "hash_locked_validation_copy"))
                options.append((folder / "branches" / branch / "best_floor_1000.pt", compact_hash, "hash_locked_validation_copy"))
        # A fixed last update may also have an intermediate snapshot. Its
        # metadata and saved validation metrics are checked before any test.
        if relative.name == "last.pt" and entry.get("expected_step") is not None:
            for folder in study_roots:
                options.append((folder / "branches" / relative.parent.name / f"step_{entry['expected_step']:07d}.pt",
                                None, "same_update_snapshot"))
        for source, wanted, recovery in options:
            if not source.is_file() or not source.stat().st_size:
                continue
            # An existing original is checked strictly in preflight. Alternatives
            # with frozen hashes must match before they can be used.
            if recovery != "original" and wanted and sha256(source) != wanted:
                continue
            sources[entry["name"]] = {"path": source, "expected_sha256": wanted, "recovery": recovery}
            break
        else:
            missing.append({"name": entry["name"], "family": entry["family"], "seed": entry.get("seed"),
                            "expected_step": entry.get("expected_step"), "path": str(original),
                            "reason": "checkpoint_missing_or_empty; no verified equivalent copy found"})
    return sources, missing


def inventory(base, selection, allow_missing_diagnostics=False, search_roots=()):
    base = Path(base)
    sources, missing_models = resolve_sources(base, selection, search_roots)
    required = {"sft", *(m["name"] for m in selection["models"] if m["family"] == selection["primary_family"])}
    missing = [m["path"] for m in missing_models if not allow_missing_diagnostics or m["name"] in required]
    paths = [m["path"] for m in sources.values()]
    paths.append(base / "artifacts/rm/best.pt")
    for folder in [base / "scaling_v1/data/large", base / "prepared"]:
        manifest = folder / "manifest.json"
        paths.append(manifest)
        if manifest.is_file() and manifest.stat().st_size:
            paths.extend(folder / name for name in json.loads(manifest.read_text())["file_sha256"])
    missing.extend(str(p) for p in paths if not p.is_file() or not p.stat().st_size)
    if missing:
        raise FileNotFoundError("Отсутствуют или пусты необходимые файлы:\n" + "\n".join(missing))
    return paths


def verify_state(entry, state, identity):
    spec = state["run_spec"]
    if state["model_spec"]["kind"] != "policy" or state["step"] != entry["expected_step"]:
        raise ValueError(f"Wrong model kind or step: {entry['name']}; expected policy at step {entry['expected_step']}, "
                         f"found {state['model_spec']['kind']} at step {state['step']}")
    if spec["manifest_sha256"] != identity["manifest_sha256"]:
        raise ValueError(f"Wrong data protocol: {entry['name']}")
    if entry["name"] == "sft":
        if spec["stage"] != "sft":
            raise ValueError("Expected original SFT")
        return
    if spec["code_sha256"] != entry["training_code_sha256"] or spec["seed"] != entry["seed"]:
        raise ValueError(f"Training code or seed differs: {entry['name']}")
    if spec["dependencies"] != {"init_sha256": identity["sft_sha256"], "reward_sha256": identity["rm_sha256"]}:
        raise ValueError(f"Training dependencies differ: {entry['name']}")
    if spec["settings"]["name"] != Path(entry["source_relative_to_drive_base"]).parent.name:
        raise ValueError(f"Training branch differs: {entry['name']}")
    expected_stage = "sft_control" if entry["family"].startswith(("ce03_", "ce10_")) else "rl"
    if spec["stage"] != expected_stage:
        raise ValueError(f"Training stage differs: {entry['name']}")


def verify_snapshot_metrics(entry, state):
    """A snapshot must be the frozen update, with its saved validation result."""
    metrics = state.get("selection", {}).get("metrics", {})
    for field, expected, tolerance in [("ndcg@10", "validation_ndcg@10", 1e-12),
                                       ("rm_reward@10", "validation_rm_reward@10", 1e-6)]:
        if field not in metrics or expected not in entry:
            raise ValueError(f"Snapshot lacks frozen validation metadata: {entry['name']}")
        if not np.isclose(metrics[field], entry[expected], rtol=0, atol=tolerance):
            raise ValueError(f"Snapshot validation metrics differ: {entry['name']}")


def check_source(entry, found, selection, loader, allow_missing_diagnostics=False):
    """Reject invalid diagnostics before test; never relax primary checks."""
    source = found["path"]
    state, digest = None, None
    try:
        digest = sha256(source)
        if found["expected_sha256"] and digest != found["expected_sha256"]:
            raise ValueError(f"Source hash differs from frozen choice: {source}")
        state = loader(source)
        if not isinstance(state, dict):
            raise ValueError("Checkpoint is not a state dictionary")
        verify_state(entry, state, selection["source_identity"])
        if found["recovery"] == "same_update_snapshot":
            verify_snapshot_metrics(entry, state)
    except (ValueError, KeyError, EOFError, OSError, pickle.UnpicklingError) as error:
        primary = entry["name"] == "sft" or entry["family"] == selection["primary_family"]
        if primary or not allow_missing_diagnostics:
            raise
        unavailable = {"name": entry["name"], "family": entry["family"], "seed": entry.get("seed"),
            "expected_step": entry["expected_step"], "path": str(source), "source_sha256": digest,
            "actual_step": state.get("step") if isinstance(state, dict) else None,
            "reason": "checkpoint_failed_verification_before_test", "verification_error": str(error)}
        return None, None, unavailable
    return state, digest, None


def copy_atomic(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def publish(root, backup, relative_paths):
    for relative in relative_paths:
        copy_atomic(Path(root) / relative, Path(backup) / relative)


def save_arrays(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def preflight(args):
    import torch
    import transformers
    from .data import load_manifest
    from .evaluation import compatible_reward, score_policy
    from .scaling import code_identity
    from .training import atomic_checkpoint, device, freeze, load_model, read_checkpoint, seed_all

    selection = read_selection(args.selection)
    base, root, backup = Path(args.base), Path(args.runs), Path(args.backup)
    if root.resolve() == backup.resolve() or base.resolve() in {root.resolve(), backup.resolve()}:
        raise ValueError("Final output and backup must use separate directories")
    allow_missing = getattr(args, "allow_missing_diagnostics", False)
    search_roots = getattr(args, "search_root", ()) or ()
    inventory(base, selection, allow_missing, search_roots)
    resolved, missing_models = resolve_sources(base, selection, search_roots)
    identity = selection["source_identity"]
    expected = {
        base / "scaling_v1/data/large/manifest.json": identity["manifest_sha256"],
        base / "prepared/manifest.json": identity["reward_source_manifest_sha256"],
        base / "artifacts/rm/best.pt": identity["rm_sha256"],
    }
    for path, wanted in expected.items():
        if sha256(path) != wanted:
            raise ValueError(f"Source hash differs from frozen choice: {path}")
    root.mkdir(parents=True, exist_ok=True)
    backup.mkdir(parents=True, exist_ok=True)
    for label, source in [("expanded", base / "scaling_v1/data/large"), ("rm_source", base / "prepared")]:
        manifest = load_manifest(source)
        for name in ["manifest.json", *manifest["file_sha256"]]:
            destination = root / "data" / label / name
            if not destination.exists() or sha256(destination) != sha256(source / name):
                copy_atomic(source / name, destination)
        load_manifest(root / "data" / label)
    sources = {}
    for entry in selection["models"]:
        if entry["name"] not in resolved:
            continue
        found = resolved[entry["name"]]
        source = found["path"]
        print("Проверка весов:", entry["name"], flush=True)
        state, digest, unavailable = check_source(entry, found, selection,
            lambda path: read_checkpoint(path, torch.device("cpu")), allow_missing)
        if unavailable is not None:
            missing_models.append(unavailable)
            print("Дополнительная модель не прошла проверку:", json.dumps(unavailable, ensure_ascii=False), flush=True)
            continue
        sources[entry["name"]] = {"path": str(source), "source_sha256": digest, "step": state["step"],
            "original_path": entry["source_relative_to_drive_base"], "recovery": found["recovery"],
            "verification": "frozen_file_hash" if found["expected_sha256"] else "step_seed_code_dependencies_and_recorded_source_hash"}
        if found["recovery"] == "same_update_snapshot":
            sources[entry["name"]]["verification"] += "_and_saved_validation_metrics"
        target = root / "models" / entry["name"] / "best.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        compact = {k: state[k] for k in ["model", "model_spec", "run_spec", "step"]}
        compact["selection"] = state.get("selection", {"rule": "frozen_last_optimizer_update"})
        atomic_checkpoint(target, compact)
        sources[entry["name"]]["staged_sha256"] = sha256(target)
        del state, compact
    copy_atomic(base / "artifacts/rm/best.pt", root / "models/rm/best.pt")
    signature = {"selection_sha256": sha256(args.selection), "source_identity": identity,
                 "evaluation_code_sha256": code_identity(),
                 "missing_model_names": [m["name"] for m in missing_models],
                 "source_checkpoint_sha256": {name: m["source_sha256"] for name, m in sources.items()}}
    for path in [root / "preflight.json", backup / "preflight.json"]:
        if path.exists() and json.loads(path.read_text())["identity"] != signature:
            raise ValueError("Existing final evaluation has different inputs/code; do not combine results")
    data_dir = root / "data/expanded"
    dev = device()
    if dev.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("Включите GPU runtime Colab: финальная оценка на CPU по умолчанию отключена")
    reference, _ = load_model(root / "models/sft/best.pt", dev)
    freeze(reference)
    reward, _, transfer = compatible_reward(root / "models/rm/best.pt", data_dir, dev, root / "data/rm_source")
    from .data import load_split
    validation = load_split(data_dir, "validation")
    probe = {k: v[:128] for k, v in validation.items()}
    seed_all(42)
    started = time.monotonic()
    score_policy(reference, reference, reward, probe, np.load(data_dir / "popularity.npy"), dev, 64,
                 include_rollout=False)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    result = {"identity": signature, "models": sources, "missing_models": missing_models, "reward_transfer": transfer,
        "allow_missing_diagnostics": allow_missing,
        "pilot_validation_users": len(probe["targets"]), "pilot_seconds": time.monotonic() - started,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "device": str(dev),
                        "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else None},
        "note": "No training. Full last.pt hashes are recorded, not compared with compact validation-file hashes."}
    write_json(root / "preflight.json", result)
    copy_atomic(args.selection, root / "FINAL_SELECTION.json")
    write_json(root / "availability.json", {"available_models": list(sources), "missing_models": missing_models,
        "primary_family_complete": True, "basis": "file availability and checkpoint verification before test; original frozen selection unchanged"})
    publish(root, backup, ["preflight.json", "FINAL_SELECTION.json", "availability.json"])
    print(json.dumps({"models": len(sources), "pilot_seconds": result["pilot_seconds"],
                      "missing_diagnostics": [m["name"] for m in missing_models],
                      "device": str(dev), "ready": True}, ensure_ascii=False), flush=True)


def completed_result(root, backup, name):
    relative = Path("evaluation/test")
    for folder in [root, backup]:
        marker = folder / relative / f"{name}.done.json"
        if not marker.exists():
            continue
        record = json.loads(marker.read_text())
        for filename, wanted in record["files"].items():
            path = folder / relative / filename
            if not path.is_file() or sha256(path) != wanted:
                raise ValueError(f"Incomplete or changed saved result: {path}")
            if folder != root:
                copy_atomic(path, root / relative / filename)
        if folder != root:
            copy_atomic(marker, root / relative / marker.name)
        return json.loads((root / relative / f"{name}.json").read_text())
    return None


def finish(root, backup, selection, preflight_result, data, metrics):
    from .baseline import regression_metrics
    from .metrics import paired_bootstrap

    output = root / "evaluation/test"
    with np.load(output / "sft_per_user.npz", allow_pickle=False) as z:
        sft = {k: z[k] for k in z.files}
    summary = {"split": "test", "n_users": len(data["users"]), "models": metrics,
        "manifest_sha256": selection["source_identity"]["manifest_sha256"],
        "reference_checkpoint_sha256": selection["source_identity"]["sft_sha256"],
        "reward_checkpoint_sha256": selection["source_identity"]["rm_sha256"],
        "selection_sha256": preflight_result["identity"]["selection_sha256"],
        "test_previously_examined": True, "test_driven_reselection_allowed": False,
        "paired_vs_sft": {}, "reward_transfer": preflight_result["reward_transfer"],
        "missing_models": preflight_result.get("missing_models", []),
        "all_requested_models_evaluated": not preflight_result.get("missing_models"),
        "requested_checkpoint_count": len(selection["models"]),
        "available_checkpoint_count": len(preflight_result["models"]),
        "availability_basis": "recorded before test, without changing the frozen selection"}
    fields = ["recall@10", "ndcg@10", "recall@20", "ndcg@20", "rm_reward@10"]
    for name in metrics:
        if name == "sft":
            continue
        with np.load(output / f"{name}_per_user.npz", allow_pickle=False) as z:
            np.testing.assert_array_equal(z["users"], sft["users"])
            np.testing.assert_array_equal(z["targets"], sft["targets"])
            summary["paired_vs_sft"][name] = {k: paired_bootstrap(sft[k], z[k]) for k in fields if k in z.files}
    manifest = json.loads((root / "data/expanded/manifest.json").read_text())
    predictions = {"global_mean": np.full(len(data["rewards"]), manifest["global_train_reward"]),
        "item_mean_smoothed": np.load(root / "data/expanded/item_rewards.npy")[data["targets"]],
        "reward_model": sft["observed_prediction"]}
    summary["reward_model_evaluation"] = {k: regression_metrics(v, data["rewards"]) for k, v in predictions.items()}
    write_json(output / "summary.json", summary)
    rows = []
    families = {m["name"]: m["family"] for m in selection["models"]}
    for name, m in metrics.items():
        rows.append({"model": name, "family": families.get(name, name), "step": m.get("checkpoint_step"),
            **m["ranking"], "rm_reward@10": m.get("rm_reward@10"),
            "coverage": m["behavior@10"]["coverage"], "seen_fraction": m["behavior@10"]["seen_fraction"]})
    table = root / "reports/results.csv"
    table.parent.mkdir(exist_ok=True)
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for family in dict.fromkeys(m["family"] for m in selection["models"] if "seed" in m):
        family_rows = [r for r in rows if r["family"] == family]
        expected_seeds = [m["seed"] for m in selection["models"] if m["family"] == family]
        available_seeds = [m["seed"] for m in selection["models"] if m["family"] == family and m["name"] in metrics]
        complete = sorted(available_seeds) == sorted(expected_seeds)
        row = {"family": family, "n_seeds": len(family_rows), "expected_n_seeds": len(expected_seeds),
               "complete_family": complete, "available_seeds": "/".join(map(str, available_seeds)),
               "missing_seeds": "/".join(map(str, sorted(set(expected_seeds) - set(available_seeds))))}
        for field in ["ndcg@10", "recall@10", "rm_reward@10"]:
            values = np.array([r[field] for r in family_rows])
            reference = metrics["sft"]["ranking"][field] if field != "rm_reward@10" else metrics["sft"][field]
            # Do not publish a surviving-seed average as the family result.
            row.update({field + "_mean": float(values.mean()) if complete else None,
                        field + "_std": float(values.std(ddof=1)) if complete and len(values) > 1 else None,
                        field + "_change_pct_mean": float(100 * (values.mean() / reference - 1)) if complete else None})
        aggregates.append(row)
    with (root / "reports/seed_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    publish(root, backup, ["evaluation/test/summary.json", "reports/results.csv", "reports/seed_summary.csv"])


def run(args):
    import torch
    from .baseline import popularity_evaluation
    from .budget import exclusive_run
    from .data import load_manifest, load_split
    from .evaluation import compare_real_and_synthetic_prefixes, compatible_reward, score_policy
    from .scaling import code_identity
    from .training import device, freeze, load_model, seed_all

    root, backup = Path(args.runs), Path(args.backup)
    if not (root / "preflight.json").exists():
        raise RuntimeError("Сначала выполните preflight; проверка и копирование весов обязательны")
    checked = json.loads((root / "preflight.json").read_text())
    selection = read_selection(root / "FINAL_SELECTION.json")
    if sha256(root / "FINAL_SELECTION.json") != checked["identity"]["selection_sha256"] or code_identity() != checked["identity"]["evaluation_code_sha256"]:
        raise ValueError("Frozen selection or evaluation code changed after preflight")
    for name, source in checked["models"].items():
        if sha256(root / "models" / name / "best.pt") != source["staged_sha256"]:
            raise ValueError(f"Staged checkpoint changed: {name}")
    if sha256(root / "models/rm/best.pt") != selection["source_identity"]["rm_sha256"]:
        raise ValueError("Staged reward model changed")
    data_dir = root / "data/expanded"
    for path, expected in [(data_dir / "manifest.json", "manifest_sha256"),
                           (root / "data/rm_source/manifest.json", "reward_source_manifest_sha256")]:
        if sha256(path) != selection["source_identity"][expected]:
            raise ValueError(f"Staged manifest changed: {path}")
    manifest = load_manifest(data_dir)
    data = load_split(data_dir, "test")
    prefixes = load_split(data_dir, "test_real_prefixes")
    dev = device()
    if dev.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("Для финальной оценки включите GPU runtime")
    reference, _ = load_model(root / "models/sft/best.pt", dev)
    freeze(reference)
    reward, _, _ = compatible_reward(root / "models/rm/best.pt", data_dir, dev, root / "data/rm_source")
    popularity = np.load(data_dir / "popularity.npy")
    jobs = [*(m for m in selection["models"] if m["name"] in checked["models"]),
            *({"name": f"rerank_{v:g}", "family": "rerank", "lambda": v} for v in selection["rerank_lambdas"]),
            {"name": "popularity", "family": "popularity"}]
    output = root / "evaluation/test"
    output.mkdir(parents=True, exist_ok=True)
    previous = [json.loads(p.read_text()) for p in [backup / "progress.json", root / "progress.json"] if p.exists()]
    if any(p["identity"] != checked["identity"] for p in previous):
        raise ValueError("Saved progress belongs to another evaluation")
    elapsed_before = max((p.get("elapsed_seconds", 0.) for p in previous), default=0.)
    started = time.monotonic()
    deadline = started + args.max_minutes * 60 - elapsed_before
    results = {}
    with exclusive_run(root):
        try:
            for job in jobs:
                name = job["name"]
                cached = completed_result(root, backup, name)
                if cached is not None:
                    results[name] = cached
                    print("Восстановлена оценка:", name, flush=True)
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("Лимит финальной оценки исчерпан; завершенные модели сохранены")
                print(f"Test evaluation {len(results)+1}/{len(jobs)}: {name}", flush=True)
                began = time.monotonic()
                model_deadline = min(deadline, began + args.model_minutes * 60)
                if name == "popularity":
                    metrics, values, slate = popularity_evaluation(data, popularity)
                    predicted = None
                else:
                    is_rerank = job["family"] == "rerank"
                    if name == "sft" or is_rerank:
                        model, state = reference, None
                    else:
                        model, state = load_model(root / "models" / name / "best.pt", dev)
                        freeze(model)
                    seed_all(manifest["config"]["seed"])
                    rollout = name == "sft" or job["family"].endswith("_selected")
                    metrics, values, predicted, slate = score_policy(model, reference, reward, data, popularity, dev,
                        manifest["config"]["eval_batch_size"], job.get("lambda"), include_rollout=rollout,
                        deadline=model_deadline)
                    if rollout:
                        metrics["real_vs_synthetic_context"] = compare_real_and_synthetic_prefixes(model, reference, reward,
                            prefixes, dev, manifest["config"]["eval_batch_size"], manifest["config"]["seed"], deadline=model_deadline)
                    metrics["checkpoint_step"] = job.get("expected_step")
                    metrics["source_checkpoint_sha256"] = checked["models"]["sft" if is_rerank else name]["source_sha256"]
                    if is_rerank:
                        metrics["lambda"] = job["lambda"]
                    if model is not reference:
                        del model, state
                metrics["elapsed_seconds"] = time.monotonic() - began
                arrays = {"users": data["users"], "targets": data["targets"], "slate": slate, **values}
                if predicted is not None:
                    arrays["observed_prediction"] = predicted
                save_arrays(output / f"{name}_per_user.npz", **arrays)
                write_json(output / f"{name}.json", metrics)
                files = [f"{name}.json", f"{name}_per_user.npz"]
                write_json(output / f"{name}.done.json", {"files": {f: sha256(output / f) for f in files}})
                publish(root, backup, [Path("evaluation/test") / f for f in [*files, f"{name}.done.json"]])
                results[name] = metrics
                print(f"Сохранено: {name}; {metrics['elapsed_seconds']:.1f} сек.", flush=True)
            finish(root, backup, selection, checked, data, results)
        finally:
            available_complete = len(results) == len(jobs) and (output / "summary.json").is_file()
            primary = {m["name"] for m in selection["models"] if m["family"] == selection["primary_family"]}
            write_json(root / "progress.json", {"identity": checked["identity"], "completed": list(results),
                "elapsed_seconds": elapsed_before + time.monotonic() - started,
                "complete": available_complete and not checked.get("missing_models"),
                "available_models_complete": available_complete,
                "primary_complete": {"sft", *primary}.issubset(results),
                "missing_models": checked.get("missing_models", []),
                "scheduled_evaluations": len(jobs),
                "requested_evaluations": len(selection["models"]) + len(selection["rerank_lambdas"]) + 1})
            publish(root, backup, ["progress.json"])


def export(args):
    root, backup = Path(args.runs), Path(args.backup)
    archive = root.parent / "results_final_test_v1.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for folder in ["evaluation", "reports"]:
            source = backup / folder if (backup / folder).exists() else root / folder
            for path in sorted(source.rglob("*")):
                if path.is_file() and path.suffix in {".json", ".npz", ".csv"}:
                    bundle.write(path, "runs/" + str(path.relative_to(source.parent)))
        for name in ["preflight.json", "FINAL_SELECTION.json", "availability.json", "progress.json"]:
            path = backup / name if (backup / name).exists() else root / name
            if path.exists():
                bundle.write(path, "runs/" + name)
        for label in ["expanded", "rm_source"]:
            for name in ["manifest.json", "vocab.json"]:
                path = root / "data" / label / name
                if path.exists():
                    bundle.write(path, "data/" + label + "/" + name)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Export archive is damaged")
    copy_atomic(archive, backup / archive.name)
    print("Архив результатов:", archive, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["inventory", "preflight", "run", "export"], required=True)
    parser.add_argument("--base", default="/content/drive/MyDrive/tbank_rl_recsys")
    parser.add_argument("--runs", default="/content/tbanktask_scaling/artifacts_final_test_v1")
    parser.add_argument("--backup", default="/content/drive/MyDrive/tbank_rl_recsys/final_test_v1")
    parser.add_argument("--selection", default="reports/rl_expectation_v1/FINAL_SELECTION.json")
    parser.add_argument("--max-minutes", type=float, default=60.)
    parser.add_argument("--model-minutes", type=float, default=5.)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--allow-missing-diagnostics", action="store_true",
                        help="Record unavailable diagnostics; SFT and every primary seed remain mandatory")
    parser.add_argument("--search-root", action="append", default=[], help="Previous local Colab project directory")
    args = parser.parse_args()
    if args.max_minutes <= 0 or args.model_minutes <= 0:
        parser.error("Time limits must be positive")
    if args.stage == "inventory":
        selection = read_selection(args.selection)
        paths = inventory(args.base, selection, args.allow_missing_diagnostics, args.search_root)
        sources, missing = resolve_sources(args.base, selection, args.search_root)
        print(json.dumps({"required_files_available": len(paths), "available_checkpoints": len(sources),
                          "missing_diagnostics": missing}, ensure_ascii=False, indent=2))
    else:
        {"preflight": preflight, "run": run, "export": export}[args.stage](args)


if __name__ == "__main__":
    main()
