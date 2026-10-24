import argparse
import json
from pathlib import Path

from .data import audit, prepare, read_interactions
from .io import download_kion, sha256, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Budget sequential recommendation experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--output", default="data/raw")
    download.add_argument("--revision", help="Full commit SHA; otherwise resolve main once")
    check = sub.add_parser("audit")
    check.add_argument("--input", required=True)
    check.add_argument("--output", default="reports/data_audit.json")
    prep = sub.add_parser("prepare")
    prep.add_argument("--input", required=True)
    prep.add_argument("--output", default="data/prepared")
    prep.add_argument("--config", default="configs/budget.json")
    prep.add_argument("--cutoffs", nargs=2, metavar=("T1", "T2"))
    baseline = sub.add_parser("baseline")
    baseline.add_argument("--data", default="data/prepared")
    baseline.add_argument("--split", choices=["validation", "test"], default="validation")
    baseline.add_argument("--output", default="reports/popularity.json")
    for name in ("train", "evaluate", "report"):
        item = sub.add_parser(name)
        item.add_argument("--data", default="data/prepared")
        item.add_argument("--runs", default="artifacts")
        if name == "train":
            item.add_argument("--stage", choices=["sft", "rm", "rl"], required=True)
            item.add_argument("--name", help="Run directory name; default is stage name")
            item.add_argument("--resume", action="store_true")
            item.add_argument("--backup", help="Mirror checkpoints to this directory, e.g. Google Drive")
            item.add_argument("--init", default="sft", help="SFT run for RL initialization/reference")
            item.add_argument("--reward-model", default="rm")
            item.add_argument("--beta", type=float, default=.03)
            item.add_argument("--repeat-penalty", type=float, default=0.)
            item.add_argument("--horizon", type=int, choices=[1, 5], default=1)
            item.add_argument("--max-steps", type=int)
            item.add_argument("--max-minutes", type=float)
        elif name == "evaluate":
            item.add_argument("--split", choices=["validation", "test"], default="validation")
            item.add_argument("--names", nargs="+", default=["sft"])
            item.add_argument("--reward-model", default="rm")
            item.add_argument("--reward-source-data", help="Original prepared data, required for explicit transfer of an unchanged RM")
            item.add_argument("--rerank-lambda", type=float, default=1.)
        else:
            item.add_argument("--output", default="reports/results")
    args = parser.parse_args(argv)
    if args.command == "download":
        print(download_kion(args.output, args.revision))
    elif args.command == "audit":
        result = audit(read_interactions(args.input))
        result["source_csv_sha256"] = sha256(args.input)
        write_json(args.output, result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "prepare":
        result = prepare(args.input, args.output, json.loads(Path(args.config).read_text()), args.cutoffs)
        print(json.dumps({k: result[k] for k in ("examples", "vocab_size", "t1", "t2", "preparation_stats")}, indent=2))
    elif args.command == "baseline":
        import numpy as np
        from .baseline import popularity_evaluation
        from .data import load_split
        result, _, _ = popularity_evaluation(load_split(args.data, args.split), np.load(Path(args.data) / "popularity.npy"))
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
    else:
        # CPU data preparation remains usable without torch/transformers.
        if args.command == "train":
            from .training import train
            train(args)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            evaluate(args)
        else:
            from .reporting import report
            report(args)


if __name__ == "__main__":
    main()
