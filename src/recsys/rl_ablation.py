"""Compare CE/KL controls with frozen exported RL results, using only CPU."""

import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .io import sha256, write_json
from .metrics import paired_bootstrap
from .rl_frontier import fraction_tag, study_fractions


def validate_reference(archive, plan, identity):
    if sha256(archive) != plan["comparison_archive_sha256"]:
        raise ValueError("Reference results archive hash differs")
    with ZipFile(archive) as z:
        previous = json.loads(z.read("runs/experiment.json"))
        summary = json.loads(z.read("runs/evaluation/validation/summary.json"))
    for key in ["sft_sha256", "rm_sha256", "manifest_sha256", "reward_source_manifest_sha256"]:
        if previous["identity"][key] != identity[key]:
            raise ValueError(f"Reference study differs: {key}")
    for key in ["reference_checkpoint_sha256", "reward_checkpoint_sha256", "manifest_sha256"]:
        expected = {"reference_checkpoint_sha256": "sft_sha256", "reward_checkpoint_sha256": "rm_sha256",
                    "manifest_sha256": "manifest_sha256"}[key]
        if summary[key] != identity[expected]:
            raise ValueError(f"Reference evaluation differs: {key}")
    original = previous["identity"]["plan"]
    for key in ["relevance_fraction", "selection_fractions", "seed"]:
        if original[key] != plan[key]:
            raise ValueError(f"Reference selection differs: {key}")
    old_branches = {b["name"]: b for b in original["branches"]}
    ignored = {"name", "family", "reference_branch", "reinforce_weight"}
    comparison_kind = plan.get("comparison_kind", "ce_kl_control")
    if comparison_kind == "estimator_study":
        ignored |= {"objective_mode", "catalog_chunk_size", "beta", "max_minutes"}
    elif comparison_kind != "ce_kl_control":
        raise ValueError("Unknown historical comparison kind")
    for branch in plan["branches"]:
        old_name = branch["reference_branch"]
        old = {**original["training"], **old_branches[old_name]}
        new = {**plan["training"], **branch}
        if {k: v for k, v in old.items() if k not in ignored} != {k: v for k, v in new.items() if k not in ignored}:
            raise ValueError(f"Training settings differ beyond the ablation: {old_name}")
        expected_weight = 1. if comparison_kind == "estimator_study" else 0.
        if old.get("reinforce_weight", 1.) != 1. or new.get("reinforce_weight", 1.) != expected_weight:
            raise ValueError("Unexpected REINFORCE setting for this comparison")
    return previous, summary


def compare_exports(root, archive, plan):
    root = Path(root)
    journal = json.loads((root / "experiment.json").read_text())
    previous, old_summary = validate_reference(archive, plan, journal["identity"])
    current = json.loads((root / "evaluation/validation/summary.json").read_text())
    if current["split"] != "validation" or current["n_users"] != old_summary["n_users"]:
        raise ValueError("Evaluation population differs")
    ref_ndcg = current["models"]["sft"]["ranking"]["ndcg@10"]
    ref_reward = current["models"]["sft"]["rm_reward@10"]
    rows, missing = [], []
    with ZipFile(archive) as bundle:
        def old_values(name):
            with np.load(io.BytesIO(bundle.read(f"runs/evaluation/validation/{name}_per_user.npz")), allow_pickle=False) as z:
                return {k: z[k] for k in z.files}

        def new_values(name):
            with np.load(root / "evaluation/validation" / (name + "_per_user.npz"), allow_pickle=False) as z:
                return {k: z[k] for k in z.files}

        before, after = old_values("sft"), new_values("sft")
        for key in ["users", "targets", "slate", "ndcg@10", "rm_reward@10"]:
            if key == "rm_reward@10":
                np.testing.assert_allclose(before[key], after[key], atol=1e-6, rtol=0)
            else:
                np.testing.assert_array_equal(before[key], after[key])
        for branch in plan["branches"]:
            name, old_name = branch["name"], branch["reference_branch"]
            if name not in current.get("selection_by_constraint", {}):
                missing.append(name)
                continue
            pairs = []
            for fraction in study_fractions(plan):
                key = fraction_tag(fraction)
                pairs.append((key, current["selection_by_constraint"][name][key]["model"],
                              old_summary["selection_by_constraint"][old_name][key]["model"]))
            new_last = current["last_checkpoint_aliases"][name]
            old_last = old_summary["last_checkpoint_aliases"][old_name]
            expected_steps = plan["training"]["max_steps"]
            if current["models"][new_last]["checkpoint_step"] == old_summary["models"][old_last]["checkpoint_step"] == expected_steps:
                pairs.append((f"fixed_{expected_steps}", new_last, old_last))
            else:
                missing.append(name + "_fixed_steps")
            for selection, control, rl in pairs:
                c, r = new_values(control), old_values(rl)
                for values in [c, r]:
                    np.testing.assert_array_equal(values["users"], after["users"])
                    np.testing.assert_array_equal(values["targets"], after["targets"])
                c_ndcg, r_ndcg = float(c["ndcg@10"].mean()), float(r["ndcg@10"].mean())
                c_reward, r_reward = float(c["rm_reward@10"].mean()), float(r["rm_reward@10"].mean())
                rows.append({"family": branch["family"], "seed": branch["seed"], "selection": selection,
                    "control": control, "rl": rl,
                    "control_step": current["models"][control]["checkpoint_step"],
                    "rl_step": old_summary["models"][rl]["checkpoint_step"],
                    "control_ndcg_change_pct": 100 * (c_ndcg / ref_ndcg - 1),
                    "rl_ndcg_change_pct": 100 * (r_ndcg / ref_ndcg - 1),
                    "control_reward_change_pct": 100 * (c_reward / ref_reward - 1),
                    "rl_reward_change_pct": 100 * (r_reward / ref_reward - 1),
                    "paired_rl_minus_control": {metric: paired_bootstrap(c[metric], r[metric])
                                                for metric in ["ndcg@10", "rm_reward@10"]}})
    aggregates = []
    for family, selection in sorted({(r["family"], r["selection"]) for r in rows}):
        values = [r for r in rows if r["family"] == family and r["selection"] == selection]
        row = {"family": family, "selection": selection, "n_seeds": len(values)}
        for metric in ["control_ndcg_change_pct", "rl_ndcg_change_pct", "control_reward_change_pct", "rl_reward_change_pct"]:
            numbers = np.array([v[metric] for v in values])
            row[metric + "_mean"] = float(numbers.mean())
            row[metric + "_std"] = float(numbers.std(ddof=1)) if len(numbers) > 1 else None
        for metric in ["ndcg@10", "rm_reward@10"]:
            deltas = [v["paired_rl_minus_control"][metric]["mean_delta"] for v in values]
            row[metric + "_rl_minus_control_mean"] = float(np.mean(deltas))
        aggregates.append(row)
    estimator_study = plan.get("comparison_kind") == "estimator_study"
    result = {"reference_archive_sha256": sha256(archive),
        "reference_code_sha256": previous["identity"]["code_sha256"],
        "control_code_sha256": journal["identity"]["code_sha256"],
        "comparisons": rows, "seed_aggregates": aggregates, "missing_or_incomplete": missing,
        "note": "RL minus CE/KL control; same frozen SFT/RM/data, matched seeds. Fixed-step comparisons isolate training duration; threshold comparisons include checkpoint selection. User bootstrap is post-selection and does not account for training variability or multiplicity."}
    output = root / "reports"
    if estimator_study:
        return write_estimator_report(output, result)
    write_json(output / "ablation_comparison.json", result)
    if aggregates:
        with (output / "ablation_seed_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
            writer.writeheader()
            writer.writerows(aggregates)
    lines = [("# Сравнение нового RL с прежним sampled REINFORCE" if estimator_study else "# Вклад REINFORCE: сравнение с SFT-loss + KL"), "",
             ("Колонки «контроль» — новый кандидат, «RL» — прежний sampled REINFORCE. Новые семейства меняют оценку градиента и/или beta; лимит ветки 8 вместо 12 минут." if estimator_study
              else "Контроль обучается без REINFORCE; исходные веса, данные, optimizer и seeds совпадают."), "",
             "| Семейство | Выбор | Seeds | ΔNDCG контроля | ΔNDCG RL | Δreward контроля | Δreward RL |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in aggregates:
        lines.append(f"| {r['family']} | {r['selection']} | {r['n_seeds']} | {r['control_ndcg_change_pct_mean']:+.3f}% | {r['rl_ndcg_change_pct_mean']:+.3f}% | {r['control_reward_change_pct_mean']:+.3f}% | {r['rl_reward_change_pct_mean']:+.3f}% |")
    lines += ["", "Все изменения относительны к SFT. Парные пользовательские интервалы и данные каждого seed — в ablation_comparison.json.",
              "Выбор по validation может быть оптимистичен. Интервалы не доказывают неухудшаемость, стандартное отклонение по seeds не является доверительным интервалом.",
              "При fixed_2500 сравниваются последние состояния после одинакового числа обновлений; при floor_* — выбранные checkpoints, которые могут быть на разных шагах.",
              "Архив с прежним RL закреплен hash; новые веса обучаются отдельно. Test не используется.", ""]
    if missing:
        lines += ["Неполные сравнения: " + ", ".join(missing), ""]
    (output / "ablation.md").write_text("\n".join(lines))
    return result


def write_estimator_report(output, comparison):
    """Use candidate-minus-historical columns for the new estimator experiment."""
    points, aggregates = [], []
    for row in comparison["comparisons"]:
        point = {k: row[k] for k in ["family", "seed", "selection"]}
        for key, value in row.items():
            if key not in point and key != "paired_rl_minus_control":
                renamed = "candidate" + key[len("control"):] if key.startswith("control") else "historical" + key[2:] if key.startswith("rl") else key
                point[renamed] = value
        point["paired_candidate_minus_historical"] = {metric: {
            "mean_delta": -values["mean_delta"], "ci95": [-values["ci95"][1], -values["ci95"][0]]}
            for metric, values in row["paired_rl_minus_control"].items()}
        points.append(point)
    for row in comparison["seed_aggregates"]:
        converted = {k: row[k] for k in ["family", "selection", "n_seeds"]}
        for key, value in row.items():
            if key.startswith("control_"):
                converted["candidate_" + key[len("control_"):]] = value
            elif key.startswith("rl_"):
                converted["historical_" + key[len("rl_"):]] = value
            elif key.endswith("_rl_minus_control_mean"):
                converted[key.replace("_rl_minus_control_mean", "_candidate_minus_historical_mean")] = -value
        aggregates.append(converted)
    result = {"reference_archive_sha256": comparison["reference_archive_sha256"],
        "historical_code_sha256": comparison["reference_code_sha256"],
        "candidate_code_sha256": comparison["control_code_sha256"], "comparisons": points,
        "seed_aggregates": aggregates, "missing_or_incomplete": comparison["missing_or_incomplete"],
        "note": "Candidate minus historical sampled REINFORCE. Same SFT/RM/data and matched seeds; estimator and/or beta differ. New branch limit is 8 rather than 12 minutes. Fixed endpoints compared only if update counts match. Post-selection user bootstrap excludes training variability and multiplicity."}
    output = Path(output)
    write_json(output / "historical_comparison.json", result)
    if aggregates:
        with (output / "historical_seed_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
            writer.writeheader()
            writer.writerows(aggregates)
    lines = ["# Новый RL и прежний sampled REINFORCE", "",
             "Разности в JSON: новый кандидат минус прежний REINFORCE. Все изменения в таблице относительны к SFT.", "",
             "| Семейство | Выбор | Seeds | ΔNDCG нового | ΔNDCG прежнего | Δreward нового | Δreward прежнего |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in aggregates:
        lines.append(f"| {r['family']} | {r['selection']} | {r['n_seeds']} | {r['candidate_ndcg_change_pct_mean']:+.3f}% | {r['historical_ndcg_change_pct_mean']:+.3f}% | {r['candidate_reward_change_pct_mean']:+.3f}% | {r['historical_reward_change_pct_mean']:+.3f}% |")
    lines += ["", "Новые семейства меняют способ оценки градиента и/или KL beta. Лимит ветки — 8 вместо 12 минут.",
              "Фиксированный шаг 2500 сравнивается только при его достижении. Пороги используют выбранные checkpoints на разных шагах.",
              "Все seeds используют одних пользователей. Bootstrap после выбора checkpoint может быть оптимистичен; неухудшаемость не доказана.", ""]
    if result["missing_or_incomplete"]:
        lines += ["Неполные сравнения: " + ", ".join(result["missing_or_incomplete"]), ""]
    (output / "historical.md").write_text("\n".join(lines))
    return result
