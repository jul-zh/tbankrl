"""CPU audit and strictly temporal examples; no torch import is required."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import sha256, write_json

PAD, BOS, UNK, FIRST_ITEM = 0, 1, 2, 3
REQUIRED = {"user_id", "item_id", "last_watch_dt", "watched_pct"}


def read_interactions(path):
    frame = pd.read_csv(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing KION columns: {sorted(missing)}")
    frame = frame[list(sorted(REQUIRED | ({"total_dur"} & set(frame.columns))))].copy()
    for key in ("user_id", "item_id"):
        values = pd.to_numeric(frame[key], errors="coerce")
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f"{key} must contain nonnegative integer IDs without missing values")
        frame[key] = values.astype("int64")
    frame["timestamp"] = pd.to_datetime(frame.pop("last_watch_dt"), errors="coerce", utc=True)
    if frame.timestamp.isna().any():
        raise ValueError("Invalid/missing last_watch_dt: repair the source before splitting")
    frame["watched_pct"] = pd.to_numeric(frame.watched_pct, errors="coerce")
    frame.loc[~np.isfinite(frame.watched_pct), "watched_pct"] = np.nan
    frame["reward"] = (frame.watched_pct / 100).clip(0, 1)
    if "total_dur" in frame:
        frame["total_dur"] = pd.to_numeric(frame.total_dur, errors="coerce")
    if frame.empty:
        raise ValueError("Empty interactions file")
    return frame


def audit(frame):
    lengths = frame.groupby("user_id").size()
    times = frame.groupby(["user_id", "timestamp"]).size()
    reward = frame.reward.dropna()
    warnings = [
        "last_watch_dt is a last-observed date, not a verified event stream. Results use an ordered-row proxy.",
        "total_dur is observed user feedback; it is never used as candidate duration or an input feature.",
        "No exposure propensities: reward predictions cannot establish causal policy improvement.",
    ]
    if (times > 1).any():
        warnings.append("Tied user timestamps have unknown order: skip their targets, encode their history as UNK.")
    if frame.duplicated(["user_id", "item_id"]).any():
        warnings.append("Repeated user-item pairs exist; inspect aggregation semantics before interpreting next-item results.")
    return {
        "rows": len(frame), "users": int(lengths.size), "items": int(frame.item_id.nunique()),
        "min_time": str(frame.timestamp.min()), "max_time": str(frame.timestamp.max()),
        "unique_timestamps": int(frame.timestamp.nunique()),
        "all_timestamps_at_midnight": bool((frame.timestamp == frame.timestamp.dt.normalize()).all()),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_user_item_rows": int(frame.duplicated(["user_id", "item_id"]).sum()),
        "tied_user_time_groups": int((times > 1).sum()),
        "rows_in_tied_groups": int(times[times > 1].sum()),
        "history_length_quantiles": {str(k): float(v) for k, v in lengths.quantile([0, .25, .5, .75, .9, .99, 1]).items()},
        "users_min_history": {str(k): int((lengths >= k).sum()) for k in (3, 5, 10)},
        "reward_missing_rows": int(frame.reward.isna().sum()),
        "reward_clipped_rows": int(((frame.watched_pct < 0) | (frame.watched_pct > 100)).sum()),
        "reward_mean": float(reward.mean()) if len(reward) else None,
        "reward_quantiles": {str(k): float(v) for k, v in reward.quantile([0, .25, .5, .75, 1]).items()} if len(reward) else {},
        "warnings": warnings,
    }


def temporal_cutoffs(frame, train_fraction=.8, validation_fraction=.1):
    if not 0 < train_fraction < train_fraction + validation_fraction < 1:
        raise ValueError("Require 0 < train_fraction < train_fraction + validation_fraction < 1")
    # Quantiles of unique times, not rows: one date cannot cross a boundary.
    times = frame.timestamp.sort_values().unique()
    if len(times) < 3:
        raise ValueError("At least three distinct timestamps are required")
    first = min(max(1, int(len(times) * train_fraction)), len(times) - 2)
    second = min(max(first + 1, int(len(times) * (train_fraction + validation_fraction))), len(times) - 1)
    return pd.Timestamp(times[first]), pd.Timestamp(times[second])


def select_users(train, minimum, limit, seed):
    counts = train.groupby("user_id").size()
    users = counts.index[counts >= minimum].to_numpy()
    if len(users) > limit:
        users = np.random.default_rng(seed).choice(users, size=limit, replace=False)
    return np.sort(users)


def build_examples(frame, mapping, selected_users, t1, t2, context):
    """Each target is preceded only by strictly older timestamp groups.

    Unknown/tied events stay as UNK placeholders so filtering cannot fabricate
    an item-to-item transition through unobserved or unordered content.
    """
    if context < 2:
        raise ValueError("context must be at least 2")
    examples = {name: [] for name in ("train", "validation", "test")}
    stats = {"ambiguous_targets_skipped": 0, "oov_targets": {k: 0 for k in examples},
             "singleton_targets": {k: 0 for k in examples}}
    ordered = frame[frame.user_id.isin(selected_users)].sort_values(["user_id", "timestamp"], kind="stable")
    for user, group in ordered.groupby("user_id", sort=False):
        history, history_times = [], []
        # Scan arrays rather than creating a pandas group for every event.
        items = group.item_id.to_numpy()
        timestamps = group.timestamp.array.asi8 // 10**9
        rewards = group.reward.to_numpy()
        boundaries = np.r_[0, np.flatnonzero(np.diff(timestamps)) + 1, len(group)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            timestamp = int(timestamps[start])
            split = "train" if timestamp < t1.timestamp() else "validation" if timestamp < t2.timestamp() else "test"
            if end - start == 1:
                target = mapping.get(int(items[start]), UNK)
                stats["singleton_targets"][split] += 1
                if target == UNK:
                    stats["oov_targets"][split] += 1
                elif history:
                    tokens = ([BOS] + history)[-context:]
                    padded = tokens + [PAD] * (context - len(tokens))
                    examples[split].append((
                        padded, target, rewards[start], int(user), timestamp, history_times[-1]
                    ))
                history.append(target)
                history_times.append(timestamp)
            else:
                stats["ambiguous_targets_skipped"] += int(end - start)
                history.extend([UNK] * min(end - start, context))
                history_times.extend([timestamp] * min(end - start, context))
            history, history_times = history[-context:], history_times[-context:]
    return examples, stats


def as_arrays(rows, context):
    return {
        "contexts": np.asarray([r[0] for r in rows], dtype=np.int32).reshape(-1, context),
        "targets": np.asarray([r[1] for r in rows], dtype=np.int64),
        "rewards": np.asarray([r[2] for r in rows], dtype=np.float32),
        "users": np.asarray([r[3] for r in rows], dtype=np.int64),
        "timestamps": np.asarray([r[4] for r in rows], dtype=np.int64),
        "max_history_time": np.asarray([r[5] for r in rows], dtype=np.int64),
    }


def first_per_user(rows, limit, seed):
    first = {}
    for row in rows:
        first.setdefault(row[3], row)
    users = np.array(sorted(first), dtype=np.int64)
    if len(users) > limit:
        users = np.sort(np.random.default_rng(seed).choice(users, size=limit, replace=False))
    return [first[int(user)] for user in users]


def real_prefix_cohort(rows, evaluation_rows, limit=500, seed=42):
    """Five eligible real prefixes per user, restricted to the evaluation cohort."""
    allowed = {row[3] for row in evaluation_rows}
    grouped = {}
    for row in rows:
        if row[3] in allowed:
            group = grouped.setdefault(row[3], [])
            if len(group) < 5:
                group.append(row)
    users = [user for user in sorted(grouped) if len(grouped[user]) == 5]
    if len(users) > limit:
        users = np.sort(np.random.default_rng(seed).choice(users, size=limit, replace=False)).tolist()
    return [row for user in users for row in grouped[user]]


def prepare(csv_path, output, config, cutoffs=None):
    output = Path(output)
    if (output / "manifest.json").exists():
        raise ValueError("Prepared output already exists; use another directory to preserve the protocol")
    frame = read_interactions(csv_path)
    report = audit(frame)
    # Truly identical source rows carry no extra signal.
    frame = frame.drop_duplicates()
    if cutoffs:
        t1, t2 = (pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo else pd.Timestamp(t, tz="UTC") for t in cutoffs)
    else:
        t1, t2 = temporal_cutoffs(frame, config["train_fraction"], config["validation_fraction"])
    if not frame.timestamp.min() < t1 < t2 <= frame.timestamp.max():
        raise ValueError("Cutoffs must leave nonempty temporal windows")
    train = frame[frame.timestamp < t1]
    users = select_users(train, config["min_train_events"], config["max_users"], config["seed"])
    if not len(users):
        raise ValueError("No users satisfy min_train_events on train")
    # Only selected users are part of the experiment's training universe.
    train = train[train.user_id.isin(users)]
    ids = np.sort(train.item_id.unique())
    mapping = {int(item): i + FIRST_ITEM for i, item in enumerate(ids)}
    if len(ids) < 2:
        raise ValueError("At least two training items are required")
    rows, stats = build_examples(frame, mapping, users, t1, t2, config["context"])
    counts_before = {name: len(value) for name, value in rows.items()}
    if len(rows["train"]) > config["max_train_examples"]:
        indices = np.sort(np.random.default_rng(config["seed"]).choice(
            len(rows["train"]), config["max_train_examples"], replace=False))
        rows["train"] = [rows["train"][i] for i in indices]
    real_prefixes = {}
    for split in ("validation", "test"):
        full_rows = rows[split]
        rows[split] = first_per_user(full_rows, config["max_eval_users"], config["seed"])
        real_prefixes[split] = real_prefix_cohort(full_rows, rows[split], seed=config["seed"])
    if any(not value for value in rows.values()):
        raise ValueError("Empty prepared split: inspect dates, tie counts and selected users before training")
    output.mkdir(parents=True, exist_ok=True)
    for split, values in rows.items():
        arrays = as_arrays(values, config["context"])
        assert (arrays["max_history_time"] < arrays["timestamps"]).all()
        np.savez_compressed(output / f"{split}.npz", **arrays)
    for split, values in real_prefixes.items():
        np.savez_compressed(output / f"{split}_real_prefixes.npz", **as_arrays(values, config["context"]))
    counts = np.zeros(len(ids) + FIRST_ITEM, dtype=np.int64)
    mapped = train.item_id.map(mapping).to_numpy()
    np.add.at(counts, mapped, 1)
    np.save(output / "popularity.npy", counts)
    # Train-only reward baselines; no aggregate target statistics enter RM features.
    rewards = train.groupby("item_id").reward.agg(["sum", "count"])
    global_mean = train.reward.mean()
    if not np.isfinite(global_mean):
        raise ValueError("No observed train rewards")
    item_rewards = np.full(len(counts), global_mean, dtype=np.float32)
    for item, row in rewards.iterrows():
        item_rewards[mapping[int(item)]] = (row["sum"] + 10 * global_mean) / (row["count"] + 10)
    np.save(output / "item_rewards.npy", item_rewards)
    write_json(output / "vocab.json", {str(item): token for item, token in mapping.items()})
    np.save(output / "selected_users.npy", users)
    source_path = Path(csv_path).parent / "source.json"
    source = json.loads(source_path.read_text()) if source_path.exists() else {"note": "User-supplied CSV"}
    manifest = {
        "format_version": 1, "source": source, "source_csv_sha256": sha256(csv_path),
        "config": config, "t1": str(t1), "t2": str(t2),
        "vocab_size": len(counts), "selected_users": len(users),
        "train_users_before_selection": int(frame[frame.timestamp < t1].user_id.nunique()),
        "examples_before_sampling": counts_before,
        "examples": {name: len(value) for name, value in rows.items()},
        "real_prefix_cohort_users": {name: len(value) // 5 for name, value in real_prefixes.items()},
        "protocol": "first eligible warm singleton target per evaluation user; strict past; tied history becomes UNK",
        "temporal_semantics": "last-observed-date ordered-row proxy; not a verified viewing-event stream",
        "preparation_stats": stats, "audit": report,
        "global_train_reward": float(global_mean),
        "file_sha256": {p.name: sha256(p) for p in sorted(output.iterdir())
                        if p.is_file() and p.suffix not in {".tmp", ".partial"}},
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_split(directory, split, limit=None):
    with np.load(Path(directory) / f"{split}.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if limit is not None and len(data["targets"]) > limit:
        # User IDs in eval arrays are already seeded/subsampled; avoid a user-ID prefix.
        indices = np.sort(np.random.default_rng(42).choice(len(data["targets"]), limit, replace=False))
        data = {name: value[indices] for name, value in data.items()}
    return data


def load_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported prepared data format")
    for name, expected in manifest["file_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Prepared data changed after freezing the protocol: {name}")
    return manifest
