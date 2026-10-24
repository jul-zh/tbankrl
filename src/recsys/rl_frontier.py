"""CPU-only selection and aggregation for fixed relevance constraints."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import write_json


def fraction_tag(fraction):
    if not 0 < fraction <= 1 or abs(fraction * 1000 - round(fraction * 1000)) > 1e-8:
        raise ValueError("Relevance fractions must be in (0,1] with up to three decimals")
    return f"floor_{round(fraction * 1000):04d}"


def study_fractions(plan):
    fractions = sorted(set(plan.get("selection_fractions", [plan["relevance_fraction"]])), reverse=True)
    for fraction in fractions:
        fraction_tag(fraction)
    if plan["relevance_fraction"] not in fractions:
        raise ValueError("Primary relevance constraint must be included in selection_fractions")
    return fractions


def summarize_frontier(summary, plan, selections):
    reference = summary["models"]["sft"]
    ref_ndcg, ref_reward = reference["ranking"]["ndcg@10"], reference["rm_reward@10"]
    points = []
    for branch in plan["branches"]:
        for fraction in study_fractions(plan):
            chosen = selections.get(branch["name"], {}).get(fraction_tag(fraction))
            if chosen is None:
                continue
            model = summary["models"][chosen["model"]]
            ndcg, reward = model["ranking"]["ndcg@10"], model["rm_reward@10"]
            if ndcg < fraction * ref_ndcg - 1e-10:
                raise ValueError("Selected checkpoint violates its recorded relevance constraint")
            points.append({"family": branch.get("family", branch["name"]),
                "seed": branch.get("seed", plan["seed"]), "fraction": fraction,
                "allowed_ndcg_loss_pct": 100 * (1 - fraction), "model": chosen["model"],
                "step": model["checkpoint_step"], "ndcg@10": ndcg, "rm_reward@10": reward,
                "ndcg_change_pct": 100 * (ndcg / ref_ndcg - 1),
                "reward_change_pct": 100 * (reward / ref_reward - 1),
                "seen_fraction": model["behavior@10"]["seen_fraction"]})
            if "rl_updates" in model:
                points[-1]["rl_updates"] = model["rl_updates"]
    grouped = defaultdict(list)
    for point in points:
        grouped[point["family"], point["fraction"]].append(point)
    aggregates = []
    for (family, fraction), values in sorted(grouped.items(), key=lambda item: (item[0][0], -item[0][1])):
        if len({v["seed"] for v in values}) != len(values):
            raise ValueError("Duplicate seed in a comparison family")
        aggregate = {"family": family, "allowed_ndcg_loss_pct": round(100 * (1 - fraction), 6),
                     "fraction": fraction, "n_seeds": len(values), "seeds": [v["seed"] for v in values],
                     "n_reward_improved": sum(v["step"] > 0 and v["rm_reward@10"] > ref_reward + 1e-6 for v in values)}
        if all("rl_updates" in v for v in values):
            aggregate["n_with_rl_updates"] = sum(v["rl_updates"] > 0 for v in values)
            aggregate["n_reward_improved_after_rl"] = sum(v["rl_updates"] > 0 and v["rm_reward@10"] > ref_reward + 1e-6 for v in values)
        for metric in ["ndcg@10", "rm_reward@10", "ndcg_change_pct", "reward_change_pct", "seen_fraction"]:
            numbers = np.array([v[metric] for v in values])
            aggregate[metric + "_mean"] = float(numbers.mean())
            aggregate[metric + "_std"] = float(numbers.std(ddof=1)) if len(numbers) > 1 else None
        aggregates.append(aggregate)
    rerank = []
    for fraction in study_fractions(plan):
        eligible = {name: m for name, m in summary["models"].items()
                    if (name == "sft" or name.startswith("rerank"))
                    and m["ranking"]["ndcg@10"] >= fraction * ref_ndcg}
        name = max(eligible, key=lambda k: eligible[k]["rm_reward@10"])
        m = eligible[name]
        rerank.append({"fraction": fraction, "allowed_ndcg_loss_pct": round(100 * (1 - fraction), 6),
                       "model": name, "ndcg@10": m["ranking"]["ndcg@10"], "rm_reward@10": m["rm_reward@10"],
                       "reward_change_pct": 100 * (m["rm_reward@10"] / ref_reward - 1)})
    seed_pairs = []
    for point in points:
        controls = [v for v in points if v["family"] == "control" and v["seed"] == point["seed"] and v["fraction"] == point["fraction"]]
        if point["family"] != "control" and controls:
            control = controls[0]
            seed_pairs.append({"family": point["family"], "seed": point["seed"], "fraction": point["fraction"],
                               "ndcg_delta_vs_control": point["ndcg@10"] - control["ndcg@10"],
                               "rm_reward_delta_vs_control": point["rm_reward@10"] - control["rm_reward@10"]})
    return {"points": points, "seed_aggregates": aggregates, "rerank_by_constraint": rerank,
            "matched_seed_comparisons": seed_pairs,
            "note": "Constraints apply to validation point estimates. Seed standard deviations are not confidence intervals; checkpoint selection and a small number of seeds limit inference."}


def write_frontier_report(root, summary, plan, selections):
    root = Path(root)
    result = summarize_frontier(summary, plan, selections)
    output = root / "reports"
    output.mkdir(exist_ok=True)
    write_json(output / "frontier.json", result)
    for name, rows in [("thresholds", result["points"]), ("seed_summary", result["seed_aggregates"]),
                       ("rerank_frontier", result["rerank_by_constraint"])]:
        if not rows:
            continue
        with (output / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Награда при фиксированном ограничении релевантности", "",
             "NDCG ограничивается в точечной validation-оценке. Это не доказательство отсутствия потери на test или во всей популяции.", "",
             "| Семейство | Допустимая потеря NDCG | Seeds | Seeds с ростом reward | Среднее изменение NDCG | Средний рост reward |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["seed_aggregates"]:
        lines.append(f"| {row['family']} | {row['allowed_ndcg_loss_pct']:g}% | {row['n_seeds']} | {row['n_reward_improved']} | {row['ndcg_change_pct_mean']:+.2f}% | {row['reward_change_pct_mean']:+.2f}% |")
    lines += ["", "Средние/стандартные отклонения по seeds и парные сравнения с контрольным RL: frontier.json.",
              "Выбор checkpoint производится отдельно для каждого ограничения по одной и той же траектории обучения. У SFT шаг 0 сохраняется как допустимый fallback.", ""]
    (output / "frontier.md").write_text("\n".join(lines))
    return result
