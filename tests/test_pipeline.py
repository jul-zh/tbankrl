"""Exercise the actual CPU CLI with a small, explicitly synthetic dataset."""

import json

import pandas as pd

from recsys.cli import main
from recsys.data import load_manifest


def test_cpu_cli_pipeline_and_fingerprint(tmp_path, capsys):
    source = tmp_path / "interactions.csv"
    rows = [(u, 10 + (u + day) % 5, f"2021-01-{day:02d}", 20 + 10 * ((u + day) % 5))
            for u in range(8) for day in range(1, 21)]
    pd.DataFrame(rows, columns=["user_id", "item_id", "last_watch_dt", "watched_pct"]).to_csv(source, index=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"seed": 42, "context": 4, "min_train_events": 3,
        "max_users": 8, "max_train_examples": 500, "max_eval_users": 8,
        "train_fraction": .7, "validation_fraction": .15}))
    main(["audit", "--input", str(source), "--output", str(tmp_path / "audit.json")])
    prepared = tmp_path / "prepared"
    main(["prepare", "--input", str(source), "--output", str(prepared), "--config", str(config_path)])
    main(["baseline", "--data", str(prepared), "--output", str(tmp_path / "baseline.json")])
    result = json.loads((tmp_path / "baseline.json").read_text())
    assert result["ranking"]["recall@10"] == 1  # all five eligible items are in top-10
    load_manifest(prepared)
    import pytest
    (prepared / "vocab.json").write_text("{}")
    with pytest.raises(ValueError, match="changed after freezing"):
        load_manifest(prepared)
