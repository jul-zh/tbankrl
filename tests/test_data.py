import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from recsys.data import (
    BOS, PAD, UNK, as_arrays, audit, build_examples, first_per_user,
    load_split, prepare, read_interactions, real_prefix_cohort, temporal_cutoffs,
)
from recsys.io import extract_interactions, sha256, write_json


def frame(rows):
    result = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp", "reward"])
    result["timestamp"] = pd.to_datetime(result.timestamp, utc=True)
    result["watched_pct"] = result.reward * 100
    return result


def test_tied_times_cannot_be_targets_or_leak_item_order():
    data = frame([(1, 11, "2021-01-01", .2), (1, 12, "2021-01-02", .3),
                  (1, 13, "2021-01-02", .4), (1, 14, "2021-01-03", .5)])
    t1, t2 = pd.Timestamp("2021-01-03", tz="UTC"), pd.Timestamp("2021-01-04", tz="UTC")
    examples, stats = build_examples(data, {11: 3, 12: 4, 13: 5, 14: 6}, [1], t1, t2, 8)
    assert examples["train"] == []  # first event has no history, tied targets excluded
    assert stats["ambiguous_targets_skipped"] == 2
    result = as_arrays(examples["validation"], 8)
    assert result["contexts"].tolist() == [[BOS, 3, UNK, UNK, PAD, PAD, PAD, PAD]]
    assert result["targets"].tolist() == [6]
    assert (result["max_history_time"] < result["timestamps"]).all()
    reversed_ties = data.iloc[[0, 2, 1, 3]]
    other, _ = build_examples(reversed_ties, {11: 3, 12: 4, 13: 5, 14: 6}, [1], t1, t2, 8)
    assert examples == other


def test_oov_event_leaves_placeholder_in_history():
    data = frame([(1, 11, "2021-01-01", .2), (1, 99, "2021-01-02", .3), (1, 12, "2021-01-03", .4)])
    examples, stats = build_examples(data, {11: 3, 12: 4}, [1], pd.Timestamp("2021-01-02", tz="UTC"), pd.Timestamp("2021-01-03", tz="UTC"), 4)
    assert stats["oov_targets"]["validation"] == 1
    assert examples["test"][0][0] == [BOS, 3, UNK, PAD]


def test_global_cutoffs_do_not_split_dates():
    data = frame([(1, 10, f"2021-01-{i:02d}", .5) for i in range(1, 11)])
    t1, t2 = temporal_cutoffs(data)
    assert t1 < t2
    with pytest.raises(ValueError, match="three distinct"):
        temporal_cutoffs(data.iloc[:2])


def test_first_eval_target_is_per_user():
    rows = [([BOS], 3, .5, 1, 10, 9), ([BOS], 4, .8, 1, 11, 10), ([BOS], 3, .5, 2, 10, 9)]
    assert first_per_user(rows, 10, 42) == [rows[0], rows[2]]


def test_real_prefix_cohort_requires_five_targets_and_same_initial_users():
    rows = [([BOS], 3, .5, user, time, time - 1)
            for user, count in [(1, 6), (2, 4), (3, 5)] for time in range(1, count + 1)]
    evaluation = [row for row in first_per_user(rows, 10, 42) if row[3] in [1, 2]]
    cohort = real_prefix_cohort(rows, evaluation)
    assert len(cohort) == 5
    assert all(row[3] == 1 for row in cohort)
    assert cohort[0] == evaluation[0]


def test_reader_rejects_invalid_dates_and_ids(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("user_id,item_id,last_watch_dt,watched_pct\n1,2,broken,50\n")
    with pytest.raises(ValueError, match="last_watch_dt"):
        read_interactions(path)
    path.write_text("user_id,item_id,last_watch_dt,watched_pct\n1.5,2,2021-01-01,50\n")
    with pytest.raises(ValueError, match="user_id"):
        read_interactions(path)


def test_reward_clipping_and_missing_do_not_become_negative_labels(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("user_id,item_id,last_watch_dt,watched_pct\n1,2,2021-01-01,150\n1,3,2021-01-02,\n1,4,2021-01-03,inf\n")
    data = read_interactions(path)
    assert data.reward.iloc[0] == 1
    assert data.reward.iloc[1:].isna().all()
    result = audit(data)
    assert result["reward_clipped_rows"] == 1
    assert result["reward_missing_rows"] == 2


def test_prepare_uses_train_only_vocab_and_history(tmp_path):
    path = tmp_path / "interactions.csv"
    rows = [(user, 100 + i % 3, f"2021-01-{i:02d}", 50) for user in (1, 2, 3) for i in range(1, 11)]
    # New item only in held-out period, must not enter training catalog.
    rows.append((1, 999, "2021-01-11", 99))
    # Exercise manifest serialization on tied timestamps, like real KION.
    rows.extend((user, 103, "2021-01-02", 50) for user in (1, 2, 3))
    pd.DataFrame(rows, columns=["user_id", "item_id", "last_watch_dt", "watched_pct"]).to_csv(path, index=False)
    config = {"seed": 42, "min_train_events": 3, "max_users": 2,
              "context": 4, "max_train_examples": 100,
              "max_eval_users": 10, "train_fraction": .6, "validation_fraction": .2}
    out = tmp_path / "prepared"
    manifest = prepare(path, out, config, ["2021-01-07", "2021-01-09"])
    vocab = json.loads((out / "vocab.json").read_text())
    assert "999" not in vocab
    assert manifest["source_csv_sha256"] == sha256(path)
    assert manifest["preparation_stats"]["ambiguous_targets_skipped"] == 4
    for split in ("train", "validation", "test"):
        arrays = load_split(out, split)
        assert (arrays["max_history_time"] < arrays["timestamps"]).all()
        assert arrays["contexts"].shape[1] == 4
    assert len(set(load_split(out, "validation")["users"])) == 2
    with pytest.raises(ValueError, match="already exists"):
        prepare(path, out, config)
    other = tmp_path / "prepared2"
    prepare(path, other, config, ["2021-01-07", "2021-01-09"])
    for split in ("train", "validation", "test"):
        a, b = load_split(out, split), load_split(other, split)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_archive_extraction_never_trusts_paths(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../../interactions.csv", "user_id,item_id\n1,2\n")
        stream.writestr("../../unrelated.txt", "must not escape")
    destination = tmp_path / "raw" / "interactions.csv"
    extract_interactions(archive, destination)
    assert destination.read_text() == "user_id,item_id\n1,2\n"
    assert not (tmp_path / "unrelated.txt").exists()


def test_json_converts_numpy_scalars_without_hiding_invalid_values(tmp_path):
    path = tmp_path / "numbers.json"
    write_json(path, {"counts": {"ties": np.int64(4)}, "fraction": np.float32(.5), "flag": np.bool_(True)})
    assert json.loads(path.read_text()) == {"counts": {"ties": 4}, "fraction": .5, "flag": True}
    previous = path.read_text()
    with pytest.raises(TypeError):
        write_json(path, {"unsupported": object()})
    with pytest.raises(ValueError):
        write_json(path, {"nonfinite": np.float32(np.nan)})
    assert path.read_text() == previous
