"""Проверка опубликованных чисел без моделей, GPU и исходного датасета."""

import csv
import json
from pathlib import Path

import numpy as np

from recsys.io import sha256, write_json
from recsys.metrics import paired_bootstrap
from recsys.scaling import code_identity


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read(path):
    return json.loads(Path(path).read_text())


def verify():
    registry = read(RESULTS / "metadata/arrays.json")
    summary = read(RESULTS / "metadata/test_summary.json")
    plan = read(ROOT / "configs/final_selection.json")
    provenance = read(RESULTS / "metadata/provenance.json")
    assert code_identity() == provenance["evaluation_code_sha256"], "Исходные модули изменились"
    assert summary["split"] == "test" and summary["n_users"] == 5000
    assert summary["test_driven_reselection_allowed"] is False
    assert summary["test_previously_examined"] is True
    allowed = {"rank", "recall@10", "recall@20", "ndcg@10", "ndcg@20", "rm_reward@10"}
    arrays, paired_rows = {}, {}
    for label, item in registry.items():
        path = RESULTS / "metrics" / (label + ".npz")
        assert sha256(path) == item["curated_array_sha256"], path
        split, name = label.split("/")
        paired_rows.setdefault(split, item["paired_rows_sha256"])
        assert paired_rows[split] == item["paired_rows_sha256"]
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) <= allowed  # Нет ID, историй и исходных откликов.
            a = {k: z[k] for k in z.files}
        assert a["rank"].shape == (5000,) and ((a["rank"] >= 0) & (a["rank"] <= 20)).all()
        for k in [10, 20]:
            hit = (a["rank"] > 0) & (a["rank"] <= k)
            ndcg = np.zeros(5000)
            ndcg[hit] = 1 / np.log2(a["rank"][hit].astype(float) + 1)
            np.testing.assert_array_equal(a[f"recall@{k}"], hit.astype(float))
            np.testing.assert_allclose(a[f"ndcg@{k}"], ndcg, rtol=0, atol=1e-12)
        for key in ["recall@10", "ndcg@10", "recall@20", "ndcg@20"]:
            np.testing.assert_allclose(a[key].mean(), item["metrics"]["ranking"][key], rtol=0, atol=1e-12)
        if "rm_reward@10" in a:
            assert np.isfinite(a["rm_reward@10"]).all()
            assert ((a["rm_reward@10"] >= 0) & (a["rm_reward@10"] <= 1)).all()
            np.testing.assert_allclose(a["rm_reward@10"].mean(), item["metrics"]["rm_reward@10"], rtol=0, atol=1e-7)
        arrays[label] = a
    assert paired_rows["validation"] != paired_rows["test"]
    expected = {m["name"] for m in plan["models"]} | {f"rerank_{v:g}" for v in plan["rerank_lambdas"]} | {"popularity"}
    assert set(summary["models"]) == expected and len(expected) == 18
    for strength in plan["rerank_lambdas"]:
        model = summary["models"][f"rerank_{strength:g}"]
        assert model["lambda"] == strength
        assert model["source_checkpoint_sha256"] == plan["source_identity"]["sft_sha256"]
    for entry in plan["models"]:
        model = summary["models"][entry["name"]]
        assert model["checkpoint_step"] == entry["expected_step"]
        assert model["source_checkpoint_sha256"] == entry["source_file_sha256"]
    intervals = 0
    baseline = arrays["test/sft"]
    for name, checks in summary["paired_vs_sft"].items():
        for metric, saved in checks.items():
            actual = paired_bootstrap(baseline[metric], arrays["test/" + name][metric])
            np.testing.assert_allclose(actual["mean_delta"], saved["mean_delta"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(actual["ci95"], saved["ci95"], rtol=0, atol=1e-12)
            intervals += 1
    assert intervals == 84
    primary = [m["name"] for m in plan["models"] if m["family"] == plan["primary_family"]]
    assert len(primary) == 3
    primary_mean = {metric: np.stack([arrays["test/" + n][metric] for n in primary]).mean(axis=0)
                    for metric in ["ndcg@10", "recall@10", "rm_reward@10"]}
    original_audit = read(RESULTS / "metadata/test_audit.json")
    for metric, values in primary_mean.items():
        actual = paired_bootstrap(baseline[metric], values)
        saved = original_audit["primary_paired_vs_sft_after_seed_average"][metric]
        np.testing.assert_allclose(actual["mean_delta"], saved["mean_delta"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual["ci95"], saved["ci95"], rtol=0, atol=1e-12)
    with (RESULTS / "tables/test_models.csv").open() as stream:
        for row in csv.DictReader(stream):
            for metric in ["ndcg@10", "recall@10"]:
                np.testing.assert_allclose(float(row[metric]), arrays["test/" + row["model"]][metric].mean(), rtol=0, atol=1e-12)
    report = {"n_users": 5000, "evaluated_variants": len(expected), "metric_arrays_checked": len(arrays),
              "paired_bootstrap_intervals_checked": intervals, "source_code_identity_verified": True,
              "all_primary_runs_present": True, "row_order_consistent_within_split": True,
              "validation_and_test_differ": True, "no_source_ids_or_histories_in_metric_arrays": True}
    write_json(RESULTS / "metadata/verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    verify()
