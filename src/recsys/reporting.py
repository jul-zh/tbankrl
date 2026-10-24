"""Generate an honest result report only from saved evaluations."""

import json
from pathlib import Path

from .io import write_json


def report(args):
    root, output = Path(args.runs), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {p.parent.name: json.loads(p.read_text()) for p in sorted((root / "evaluation").glob("*/summary.json"))}
    if "validation" not in summaries:
        raise ValueError("Run validation evaluation before generating a report")
    validation = summaries["validation"]
    if any(summary["manifest_sha256"] != validation["manifest_sha256"] for summary in summaries.values()):
        raise ValueError("Cannot combine evaluations from different data protocols")
    for split, summary in summaries.items():
        if split == "validation":
            continue
        for name in summary["models"].keys() & validation["models"].keys():
            if summary["models"][name].get("checkpoint_sha256") != validation["models"][name].get("checkpoint_sha256"):
                raise ValueError(f"Model {name} changed between validation and {split}")
    if "sft" not in validation["models"]:
        raise ValueError("Validation must contain the sft model")
    fraction = getattr(args, "relevance_fraction", .95)
    if not 0 < fraction <= 1:
        raise ValueError("Relevance fraction must be in (0,1]")
    candidate_names = getattr(args, "candidate_names", None)
    floor = fraction * validation["models"]["sft"]["ranking"]["ndcg@10"]
    sft_reward = validation["models"]["sft"].get("rm_reward@10", float("inf"))
    feasible = {name: result for name, result in validation["models"].items()
                if (candidate_names is None or name in candidate_names)
                and result.get("stage") == "rl" and result["ranking"]["ndcg@10"] >= floor
                and result["rm_reward@10"] > sft_reward + 1e-6}
    selected = max(feasible, key=lambda name: feasible[name]["rm_reward@10"]) if feasible else None
    selection = {"selected_rl": selected, "validation_ndcg_floor": floor,
                 "criterion": f"maximum validation RM reward@10 above SFT among RL runs with NDCG@10 >= {100*fraction:g}% of SFT"}
    write_json(output / "selection.json", selection)
    lines = ["# Результаты экспериментов", "", "Числа прочитаны из сохраненных оценок; RM reward — предсказанная награда.", "",
             f"Выбранная по validation RL-ветка: **{selected or 'нет допустимой ветки'}**.", "",
             f"Правило выбора каждого checkpoint указано в JSON оценки. Здесь среди допустимых RL-веток выбирается максимальная прогнозная награда выше SFT при NDCG ≥{100*fraction:g}% от SFT.", ""]
    for split, summary in summaries.items():
        lines += [f"## {split}", "", f"Пользователей: {summary['n_users']}. Протокол: `{summary['manifest_sha256']}`.", "",
                  "| Модель | Recall@10 | NDCG@10 | RM reward@10 | Coverage@10 | Уже в контексте |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name, metrics in summary["models"].items():
            rank, beh = metrics["ranking"], metrics["behavior@10"]
            reward = f"{metrics['rm_reward@10']:.4f}" if "rm_reward@10" in metrics else "—"
            lines.append(f"| {name} | {rank['recall@10']:.4f} | {rank['ndcg@10']:.4f} | {reward} | {beh['coverage']:.4f} | {beh['seen_fraction']:.4f} |")
        lines += ["", "### Reward model", "", "| Оценка | MAE | RMSE |", "| --- | ---: | ---: |"]
        for name, metrics in summary["reward_model_evaluation"].items():
            mae = f"{metrics['mae']:.4f}" if metrics["mae"] is not None else "—"
            rmse = f"{metrics['rmse']:.4f}" if metrics["rmse"] is not None else "—"
            lines.append(f"| {name} | {mae} | {rmse} |")
        lines += [""]
    lines += ["## Ограничения", "",
              "KION используется как последовательность записей по дате последнего просмотра. Она не восстанавливает реальные показы, все просмотры и порядок внутри одной даты.", "",
              "Рост RM reward не является доказательством роста реальной удовлетворенности. Пять сгенерированных шагов используют искусственное обновление истории. Bootstrap по пользователям не учитывает случайность обучения.", "",
              "Сигнал возврата: следующее наблюдаемое взаимодействие в течение 7 дней, только при полном окне наблюдения. Это прокси активности; причинное влияние рекомендации требует онлайн-проверки.", "",
              "Расширенные метрики, доверительные интервалы и шаги генерации находятся в JSON оценок. Интерпретацию reward hacking и сдвига поведения нужно написать после просмотра реальных результатов.", ""]
    (output / "README.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, metrics in validation["models"].items():
        if "rm_reward@10" in metrics:
            x, y = metrics["ranking"]["ndcg@10"], metrics["rm_reward@10"]
            axes[0].scatter(x, y)
            axes[0].annotate(name, (x, y), fontsize=8)
        rollout = metrics.get("synthetic_rollout_5")
        if rollout:
            axes[1].plot(range(1, 6), [step["rm_reward"] for step in rollout], marker="o", label=name)
    axes[0].set(xlabel="Validation NDCG@10", ylabel="Predicted reward@10")
    axes[1].set(xlabel="Synthetic rollout step", ylabel="Predicted reward", xticks=range(1, 6))
    if axes[1].lines:
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "tradeoff_and_rollout.png", dpi=160)
    plt.close(fig)
    print(output / "README.md")
