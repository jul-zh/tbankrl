"""Selection constraints and seed aggregation must not silently relax NDCG."""

import copy

import pytest

from recsys.rl_frontier import fraction_tag, study_fractions, summarize_frontier, write_frontier_report


def frontier_fixture():
    def model(ndcg, reward, step):
        return {"ranking": {"ndcg@10": ndcg}, "rm_reward@10": reward,
                "checkpoint_step": step, "behavior@10": {"seen_fraction": .1}}
    models = {"sft": model(.1, .5, 100), "fallback": model(.1, .5, 0),
              "a42": model(.101, .55, 100), "a123": model(.1, .54, 200),
              "looser": model(.099, .6, 400), "rerank_0.1": model(.099, .57, None),
              "rerank": model(.098, .7, None)}
    plan = {"seed": 42, "relevance_fraction": 1., "selection_fractions": [1., .99], "branches": []}
    selections = {}
    for family in ["control", "anchor"]:
        for seed in [42, 123]:
            name = f"{family}_{seed}"
            plan["branches"].append({"name": name, "family": family, "seed": seed})
            selections[name] = {fraction_tag(1.): {"model": f"a{seed}" if family == "anchor" else "fallback"},
                                fraction_tag(.99): {"model": "looser"}}
    return {"models": models}, plan, selections


def test_primary_constraint_and_precision_are_validated():
    assert fraction_tag(1.) == "floor_1000"
    assert fraction_tag(.99) == "floor_0990"
    assert study_fractions({"relevance_fraction": .99, "selection_fractions": [.95, 1., .99]}) == [1., .99, .95]
    with pytest.raises(ValueError): fraction_tag(.9995)
    with pytest.raises(ValueError): study_fractions({"relevance_fraction": 1., "selection_fractions": [.95]})


def test_seed_summary_uses_constraints_and_counts_fallbacks(tmp_path):
    summary, plan, selections = frontier_fixture()
    result = write_frontier_report(tmp_path, summary, plan, selections)
    anchor = next(x for x in result["seed_aggregates"] if x["family"] == "anchor" and x["fraction"] == 1.)
    assert anchor["n_seeds"] == anchor["n_reward_improved"] == 2
    assert anchor["rm_reward@10_mean"] == pytest.approx(.545)
    assert anchor["rm_reward@10_std"] == pytest.approx(.01 / 2**.5)
    control = next(x for x in result["seed_aggregates"] if x["family"] == "control" and x["fraction"] == 1.)
    assert control["n_reward_improved"] == 0
    assert result["rerank_by_constraint"][0]["model"] == "sft"
    assert result["rerank_by_constraint"][1]["model"] == "rerank_0.1"
    assert len(result["matched_seed_comparisons"]) == 4
    assert (tmp_path / "reports/seed_summary.csv").exists()
    invalid = copy.deepcopy(summary)
    invalid["models"]["looser"]["ranking"]["ndcg@10"] = .098
    with pytest.raises(ValueError, match="violates"):
        summarize_frontier(invalid, plan, selections)


def test_schedule_summary_separates_ce_only_and_post_rl_checkpoints():
    summary, plan, selections = frontier_fixture()
    for name, model in summary["models"].items():
        model["rl_updates"] = 0 if name in {"fallback", "a42"} else 5
    result = summarize_frontier(summary, plan, selections)
    anchor = next(r for r in result["seed_aggregates"] if r["family"] == "anchor" and r["fraction"] == 1.)
    assert anchor["n_reward_improved"] == 2
    assert anchor["n_with_rl_updates"] == anchor["n_reward_improved_after_rl"] == 1
