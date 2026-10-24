"""Expand train users while preserving the original evaluation and item IDs."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .budget import exclusive_run
from .data import FIRST_ITEM, PAD, UNK, as_arrays, build_examples, load_manifest, load_split, read_interactions
from .io import sha256, write_json
from .metrics import behavior, per_user_ranking, topk


def evaluation_identity(directory):
    manifest = load_manifest(directory)
    names = ["vocab.json", "validation.npz", "test.npz",
             "validation_real_prefixes.npz", "test_real_prefixes.npz"]
    return {"t1": manifest["t1"], "t2": manifest["t2"],
            "files": {name: sha256(Path(directory) / name) for name in names}}


def context_audit(data):
    tokens = data["contexts"]
    known = (tokens >= FIRST_ITEM).sum(axis=1)
    unknown = (tokens == UNK).sum(axis=1)
    lengths = known + unknown
    return {
        "examples": len(tokens), "users": int(len(np.unique(data["users"]))),
        "unknown_history_fraction": float(unknown.sum() / max(1, lengths.sum())),
        "no_known_history_fraction": float((known == 0).mean()) if len(known) else None,
        "known_history_quantiles": {str(q): float(np.quantile(known, q)) for q in [0, .25, .5, .75, .9, 1]} if len(known) else {},
    }


def prepare_scaling(csv_path, source_data, output, settings, seed=42):
    source_data, output = Path(source_data), Path(output)
    if output.resolve().is_relative_to(source_data.resolve()) or source_data.resolve().is_relative_to(output.resolve()):
        raise ValueError("Expanded data must use a separate directory from the original protocol")
    source = load_manifest(source_data)
    raw_hash = sha256(csv_path)
    if raw_hash != source["source_csv_sha256"]:
        raise ValueError("Raw CSV differs from the original experiment; do not mix data sources")
    identity = {"format_version": 1, "source_manifest_sha256": sha256(source_data / "manifest.json"),
                "source_csv_sha256": raw_hash, "settings": settings, "seed": seed,
                "evaluation": evaluation_identity(source_data)}
    if settings["max_users"] < source["selected_users"] or settings["max_added_examples"] < 1:
        raise ValueError("Scaling must retain all original users and allow additional examples")
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output):
        journal = output / "preparation.json"
        if journal.exists():
            saved = json.loads(journal.read_text())
            if saved["identity"] != identity:
                raise ValueError("Prepared scaling study has different inputs/settings")
            for variant in ["small", "large"]:
                load_manifest(output / variant)
                if evaluation_identity(output / variant) != identity["evaluation"]:
                    raise ValueError("Frozen evaluation changed")
            return saved
        frame = read_interactions(csv_path).drop_duplicates()
        t1, t2 = pd.Timestamp(source["t1"]), pd.Timestamp(source["t2"])
        train = frame[frame.timestamp < t1].copy()
        del frame
        original_users = np.load(source_data / "selected_users.npy")
        counts = train.groupby("user_id").size()
        eligible = counts.index[counts >= source["config"]["min_train_events"]].to_numpy()
        if not np.isin(original_users, eligible).all():
            raise ValueError("Original user cohort is inconsistent with the train source")
        candidates = np.setdiff1d(eligible, original_users)
        rng = np.random.default_rng(seed)
        extra = np.sort(rng.choice(candidates, min(len(candidates), settings["max_users"] - len(original_users)), replace=False))
        if not len(extra):
            raise ValueError("No additional eligible train users: inspect max_users and eligibility")
        mapping = {int(item): token for item, token in json.loads((source_data / "vocab.json").read_text()).items()}
        rows, stats = build_examples(train, mapping, extra, t1, t2, source["config"]["context"])
        added_before_cap = len(rows["train"])
        if added_before_cap > settings["max_added_examples"]:
            indices = np.sort(rng.choice(added_before_cap, settings["max_added_examples"], replace=False))
            rows["train"] = [rows["train"][i] for i in indices]
        added = as_arrays(rows["train"], source["config"]["context"])
        del rows
        if not len(added["targets"]):
            raise ValueError("Additional users yield no usable training targets")
        base = load_split(source_data, "train")
        expanded = {key: np.concatenate([value, added[key]]) for key, value in base.items()}
        assert (expanded["max_history_time"] < expanded["timestamps"]).all()
        assert (expanded["timestamps"] < t1.timestamp()).all()
        selected = np.sort(np.concatenate([original_users, extra]))
        variants = {"small": (base, original_users), "large": (expanded, selected)}
        audit = {}
        for name, (arrays, users) in variants.items():
            destination = output / name
            if not (destination / "manifest.json").exists():
                temporary = output / (name + ".partial")
                if temporary.exists():
                    shutil.rmtree(temporary)
                temporary.mkdir()
                for filename in source["file_sha256"]:
                    shutil.copy2(source_data / filename, temporary / filename)
                manifest = copy.deepcopy(source)
                if name == "large":
                    np.savez_compressed(temporary / "train.npz", **arrays)
                    np.save(temporary / "selected_users.npy", users)
                    selected_train = train[train.user_id.isin(users)]
                    known_train = selected_train[selected_train.item_id.isin(mapping)]
                    popularity = np.zeros(source["vocab_size"], dtype=np.int64)
                    np.add.at(popularity, known_train.item_id.map(mapping).to_numpy(), 1)
                    np.save(temporary / "popularity.npy", popularity)
                    global_mean = float(known_train.reward.mean())
                    if not np.isfinite(global_mean):
                        raise ValueError("Expanded training set has no observed rewards")
                    item_mean = np.full(source["vocab_size"], global_mean, dtype=np.float32)
                    for item, row in known_train.groupby("item_id").reward.agg(["sum", "count"]).iterrows():
                        item_mean[mapping[int(item)]] = (row["sum"] + 10 * global_mean) / (row["count"] + 10)
                    np.save(temporary / "item_rewards.npy", item_mean)
                    manifest["global_train_reward"] = global_mean
                    manifest["config"]["max_users"] = settings["max_users"]
                    manifest["config"]["max_train_examples"] = len(arrays["targets"])
                    manifest["selected_users"] = len(users)
                    manifest["examples"]["train"] = len(arrays["targets"])
                    manifest["examples_before_sampling"]["train"] = len(base["targets"]) + added_before_cap
                    manifest["preparation_stats"] = {"original": source["preparation_stats"], "added_train_users": stats}
                manifest["scaling"] = {"variant": name, **identity,
                    "training": "Original train arrays retained verbatim; only extra users add targets"}
                manifest["file_sha256"] = {p.name: sha256(p) for p in sorted(temporary.iterdir()) if p.is_file()}
                write_json(temporary / "manifest.json", manifest)
                temporary.replace(destination)
            actual_manifest = load_manifest(destination)
            if actual_manifest["scaling"] != {"variant": name, **identity,
                    "training": "Original train arrays retained verbatim; only extra users add targets"}:
                raise ValueError("Existing partial preparation uses different settings")
            if evaluation_identity(destination) != identity["evaluation"]:
                raise ValueError("Frozen evaluation changed during preparation")
            audit[name] = {split: context_audit(load_split(destination, split)) for split in ["train", "validation"]}
            audit[name]["selected_train_users"] = len(users)
        result = {"identity": identity, "audit": audit, "eligible_train_users": len(eligible),
                  "added_users": len(extra), "added_examples_before_cap": added_before_cap,
                  "added_examples": len(added["targets"]), "note": "Old test already viewed; this command does not evaluate test"}
        write_json(journal, result)
        return result


def last_tokens(contexts):
    # An UNK at the last position causes popularity backoff; never jump across it.
    lengths = (contexts != PAD).sum(axis=1)
    return contexts[np.arange(len(contexts)), lengths - 1]


def transition_index(train, vocab_size):
    last = last_tokens(train["contexts"])
    valid = last >= FIRST_ITEM
    keys, counts = np.unique(last[valid] * np.int64(vocab_size) + train["targets"][valid], return_counts=True)
    owners = keys // vocab_size
    offsets = np.r_[0, np.bincount(owners, minlength=vocab_size).cumsum()]
    return offsets, keys % vocab_size, counts


def evaluate_baselines(directory, output, smoothing=20., batch_size=128, reference_popularity=None):
    directory, output = Path(directory), Path(output)
    manifest = load_manifest(directory)
    train, validation = load_split(directory, "train"), load_split(directory, "validation")
    popularity = np.load(directory / "popularity.npy")
    prior = popularity.astype(float) / popularity.sum()
    offsets, targets, counts = transition_index(train, manifest["vocab_size"])
    last = last_tokens(validation["contexts"])
    slates = []
    for start in range(0, len(last), batch_size):
        scores = np.broadcast_to(smoothing * prior, (min(batch_size, len(last) - start), len(prior))).copy()
        for row, item in enumerate(last[start:start + batch_size]):
            left, right = offsets[item:item + 2]
            scores[row, targets[left:right]] += counts[left:right]
        slates.append(topk(scores, 20))
    recommendations = {"popularity": np.repeat(topk(popularity[None, :], 20), len(last), axis=0),
                       "transition": np.concatenate(slates)}
    result = {"split": "validation", "evaluation": evaluation_identity(directory), "models": {}}
    for name, slate in recommendations.items():
        per_user = per_user_ranking(slate, validation["targets"])
        result["models"][name] = {"ranking": {k: float(v.mean()) for k, v in per_user.items()},
            "behavior@10": behavior(slate[:, :10], validation["contexts"],
                                   popularity if reference_popularity is None else reference_popularity)}
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / (name + "_per_user.npz"), users=validation["users"], slate=slate, **per_user)
    write_json(output / "baselines.json", result)
    return result


def code_identity():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--source-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/scaling.json")
    args = parser.parse_args()
    plan = json.loads(Path(args.config).read_text())
    result = prepare_scaling(args.input, args.source_data, args.output, plan["data"], plan["seed"])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
