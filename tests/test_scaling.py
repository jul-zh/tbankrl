"""The scaling comparison must change train size, never the holdout or IDs."""

import json

import numpy as np
import pandas as pd
import pytest

from recsys.data import load_manifest, load_split, prepare
from recsys.scaling import context_audit, evaluate_baselines, evaluation_identity, prepare_scaling, transition_index


def scaling_fixture(tmp_path):
    csv_path = tmp_path / "interactions.csv"
    rows = [(user, 10 + (user + day) % 6, f"2021-01-{day:02d}", 10 * ((user + day) % 10))
            for user in range(8) for day in range(1, 21)]
    rows.extend([(99, 999, "2021-01-18", 100), (99, 998, "2021-01-19", 50)])
    rows.extend((user, 50, "2021-01-02", 50) for user in range(8))
    pd.DataFrame(rows, columns=["user_id", "item_id", "last_watch_dt", "watched_pct"]).to_csv(csv_path, index=False)
    config = {"seed": 42, "context": 6, "min_train_events": 3, "max_users": 3,
              "max_train_examples": 1000, "max_eval_users": 3, "train_fraction": .5, "validation_fraction": .25,
              "eval_batch_size": 4}
    source = tmp_path / "original"
    prepare(csv_path, source, config, ["2021-01-11", "2021-01-16"])
    return csv_path, source


def test_expansion_is_nested_train_only_and_preserves_evaluation(tmp_path):
    csv_path, source = scaling_fixture(tmp_path)
    root = tmp_path / "scaled"
    result = prepare_scaling(csv_path, source, root, {"max_users": 7, "max_added_examples": 1000})
    old = load_split(source, "train")
    larger = load_split(root / "large", "train")
    assert len(larger["targets"]) > len(old["targets"])
    for key in old:
        np.testing.assert_array_equal(larger[key][:len(old[key])], old[key])
        np.testing.assert_array_equal(load_split(root / "small", "train")[key], old[key])
    assert (larger["max_history_time"] < larger["timestamps"]).all()
    assert (larger["timestamps"] < pd.Timestamp("2021-01-11", tz="UTC").timestamp()).all()
    selected = np.load(root / "large/selected_users.npy")
    assert len(selected) == 7 and 99 not in selected
    assert evaluation_identity(source) == evaluation_identity(root / "large") == evaluation_identity(root / "small")
    assert "999" not in json.loads((root / "large/vocab.json").read_text())
    assert result["audit"]["large"]["train"]["unknown_history_fraction"] > 0
    raw = pd.read_csv(csv_path)
    train = raw[(raw.last_watch_dt < "2021-01-11") & raw.user_id.isin(selected)]
    mapping = {int(k): v for k, v in json.loads((source / "vocab.json").read_text()).items()}
    expected = np.zeros(load_manifest(source)["vocab_size"], dtype=np.int64)
    np.add.at(expected, train.item_id.map(mapping).dropna().astype(int).to_numpy(), 1)
    np.testing.assert_array_equal(np.load(root / "large/popularity.npy"), expected)
    assert prepare_scaling(csv_path, source, root, {"max_users": 7, "max_added_examples": 1000}) == result
    with pytest.raises(ValueError, match="different inputs"):
        prepare_scaling(csv_path, source, root, {"max_users": 6, "max_added_examples": 1000})
    csv_path.write_text(csv_path.read_text() + "100,10,2021-01-01,10\n")
    with pytest.raises(ValueError, match="Raw CSV differs"):
        prepare_scaling(csv_path, source, root, {"max_users": 7, "max_added_examples": 1000})


def test_transition_uses_immediate_context_and_backs_off_after_unk(tmp_path):
    offsets, _, _ = transition_index({"contexts": np.array([[1, 3, 2, 0], [1, 3, 2, 0]]),
                                       "targets": np.array([4, 5])}, 6)
    assert (offsets == 0).all()  # Do not manufacture transitions 3 -> 4 / 5.
    offsets, targets, counts = transition_index({"contexts": np.array([[1, 3, 0], [1, 3, 0], [1, 4, 0]]),
                                               "targets": np.array([4, 4, 3])}, 6)
    assert targets[offsets[3]:offsets[4]].tolist() == [4]
    assert counts[offsets[3]:offsets[4]].tolist() == [2]
    csv_path, source = scaling_fixture(tmp_path)
    metrics = evaluate_baselines(source, tmp_path / "evaluation")
    assert set(metrics["models"]) == {"popularity", "transition"}
    assert metrics["split"] == "validation"
    assert not (tmp_path / "evaluation/test").exists()
    for result in metrics["models"].values():
        assert result["ranking"]["recall@10"] == 1  # seven real catalog items


def test_unknown_history_audit_does_not_count_bos_or_padding():
    result = context_audit({"contexts": np.array([[1, 3, 2, 0], [1, 2, 0, 0]]), "users": np.array([1, 2])})
    assert result["unknown_history_fraction"] == pytest.approx(2 / 3)
    assert result["no_known_history_fraction"] == .5
