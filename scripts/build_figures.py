"""Построение рисунков исследования из опубликованных таблиц и метаданных."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def read_csv(name):
    with (RESULTS / "tables" / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    FIGURES.mkdir(exist_ok=True)
    plan = json.loads((ROOT / "configs/final_selection.json").read_text())
    registry = json.loads((RESULTS / "metadata/arrays.json").read_text())
    test = json.loads((RESULTS / "metadata/test_summary.json").read_text())["models"]
    groups = {"rl_ce03_selected": "RL, α=0,3, выбранный шаг",
              "rl_ce03_fixed": "RL, α=0,3, шаг 2500",
              "rl_ce10_fixed": "RL, α=1,0, шаг 2500",
              "rl_control_fixed": "RL, α=0, шаг 2500"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True, sharey=True)
    for axis, split, title in zip(axes, ["validation", "test"], ["Проверочная выборка", "Тестовая выборка"]):
        baseline = registry[split + "/sft"]["metrics"]
        for color, (family, label) in enumerate(groups.items()):
            entries = [m for m in plan["models"] if m["family"] == family]
            vals = [(e["validation_ndcg@10"], e["validation_rm_reward@10"]) if split == "validation"
                    else (test[e["name"]]["ranking"]["ndcg@10"], test[e["name"]]["rm_reward@10"]) for e in entries]
            x = [100 * (n / baseline["ranking"]["ndcg@10"] - 1) for n, r in vals]
            y = [100 * (r / baseline["rm_reward@10"] - 1) for n, r in vals]
            axis.scatter(x, y, color=f"C{color}", s=22, alpha=.5)
            axis.scatter(np.mean(x), np.mean(y), color=f"C{color}", marker="D", s=65, label=label)
        points = [(0, 0)]
        for strength in plan["rerank_lambdas"]:
            model = registry[f"{split}/rerank_{strength:g}"]["metrics"]
            point = (100 * (model["ranking"]["ndcg@10"] / baseline["ranking"]["ndcg@10"] - 1),
                     100 * (model["rm_reward@10"] / baseline["rm_reward@10"] - 1))
            points.append(point)
            axis.annotate(f"λ={strength:g}".replace(".", ","), point, xytext=(4, 4), textcoords="offset points", fontsize=8)
        axis.plot(*zip(*points), color="black", marker="x", linestyle="--", label="Переранжирование")
        axis.scatter([0], [0], color="black", marker="s", label="Исходная модель")
        axis.axvline(0, color="gray", lw=.7)
        axis.axhline(0, color="gray", lw=.7)
        axis.set(title=title, xlabel="Изменение NDCG@10, %", ylabel="Изменение прогноза награды, %")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "tradeoff.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharex=True, sharey=True)
    calibration = read_csv("calibration.csv")
    for axis, split, title in zip(axes, ["validation", "test"], ["Проверочная выборка", "Тестовая выборка"]):
        axis.plot([0, 1], [0, 1], "--", color="gray", label="Идеальная калибровка")
        for model, label in [("item_mean_smoothed", "Среднее объекта"), ("reward_model", "Модель награды")]:
            bins = [r for r in calibration if r["split"] == split and r["model"] == model]
            axis.plot([float(r["predicted"]) for r in bins], [float(r["actual"]) for r in bins], marker="o", label=label)
        axis.set(title=title, xlabel="Средний прогноз", ylabel="Наблюдаемая доля просмотра", xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "reward_calibration.png", dpi=180)
    plt.close(fig)

    primary = [m["name"] for m in plan["models"] if m["family"] == plan["primary_family"]]
    steps = np.arange(1, 6)
    rows = read_csv("rollout.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for axis, key, scale, ylabel in [(axes[0], "rm_reward", 1, "Прогноз награды"),
                                     (axes[1], "repeat_within_rollout", 100, "Повтор сгенерированного объекта, %")]:
        baseline = [scale * float(r[key]) for r in rows if r["model"] == "sft"]
        values = np.array([[scale * float(r[key]) for r in rows if r["model"] == name] for name in primary])
        axis.plot(steps, baseline, color="black", marker="s", label="Исходная модель")
        axis.fill_between(steps, values.min(0), values.max(0), color="C0", alpha=.18, label="Диапазон трех запусков RL")
        axis.plot(steps, values.mean(0), color="C0", marker="o", label="Среднее трех запусков RL")
        axis.set(xlabel="Шаг генерации", ylabel=ylabel, xticks=steps)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "rollout.png", dpi=180)
    plt.close(fig)

    rows = read_csv("prefixes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharex=True, sharey=True)
    for axis, names, title in [(axes[0], ["sft"], "Исходная модель"), (axes[1], primary, "RL: среднее трех запусков")]:
        for mode, label, style in [("real_prefixes", "Реальная история", "-"), ("synthetic_prefixes", "Сгенерированная история", "--")]:
            values = np.array([[float(r["rm_reward"]) for r in rows if r["model"] == name and r["context"] == mode] for name in names])
            axis.plot(steps, values.mean(0), marker="o", linestyle=style, label=label)
        axis.set(title=title, xlabel="Шаг", ylabel="Прогноз награды", xticks=steps)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "prefixes.png", dpi=180)
    plt.close(fig)
    print("Построены четыре рисунка в results/figures")


if __name__ == "__main__":
    main()
