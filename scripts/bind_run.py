"""Закрепление контрольных сумм входов нового запуска до обучения RL."""

import argparse
import json
from pathlib import Path

from recsys.data import load_manifest
from recsys.io import sha256, write_json


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["mods", "preserve", "ablation"], required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--sft", required=True, type=Path)
    parser.add_argument("--rm", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    name = {"mods": "rl_study", "preserve": "rl_preserve", "ablation": "rl_ablation"}[args.study]
    config = json.loads((ROOT / "configs" / (name + ".json")).read_text())
    load_manifest(args.data)
    config.update(expected_manifest_sha256=sha256(args.data / "manifest.json"),
                  expected_sft_sha256=sha256(args.sft), expected_rm_sha256=sha256(args.rm))
    # Новый контроль сравнивается с результатами нового запуска, а не со старым архивом.
    config.pop("comparison_archive", None)
    config.pop("comparison_archive_sha256", None)
    if args.output.exists() and json.loads(args.output.read_text()) != config:
        raise ValueError("Входы изменились: используйте отдельную папку нового эксперимента")
    write_json(args.output, config)
    print(args.output)


if __name__ == "__main__":
    main()
