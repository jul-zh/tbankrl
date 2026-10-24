"""Фиксация состояний нового обучения по проверочным данным, до оценки теста."""

import argparse
import copy
import json
from pathlib import Path

from recsys.data import load_manifest
from recsys.io import sha256, write_json


ROOT = Path(__file__).resolve().parents[1]


def freeze(base, template, loader):
    base = Path(base)
    plan = copy.deepcopy(template)
    data = base / "scaling_v1/data/large"
    load_manifest(data)
    load_manifest(base / "prepared")
    sft = base / "scaling_v1/large_128/best.pt"
    rm = base / "artifacts/rm/best.pt"
    plan["source_identity"] = {"manifest_sha256": sha256(data / "manifest.json"),
        "sft_sha256": sha256(sft), "rm_sha256": sha256(rm),
        "reward_source_manifest_sha256": sha256(base / "prepared/manifest.json")}
    for entry in plan["models"]:
        path = base / entry["source_relative_to_drive_base"]
        state = loader(path)
        assert state["model_spec"]["kind"] == "policy"
        assert state["run_spec"]["manifest_sha256"] == plan["source_identity"]["manifest_sha256"]
        if entry["family"].endswith("_fixed") and state["step"] != 2500:
            raise ValueError(f"{entry['name']}: завершено {state['step']} обновлений, требуется 2500")
        entry["expected_step"] = state["step"]
        entry["source_file_sha256"] = sha256(path)
        for key in ["validation_evaluation_alias", "validation_evaluated_file_sha256", "validation_ndcg@10", "validation_rm_reward@10"]:
            entry.pop(key, None)
        if entry["name"] != "sft":
            spec = state["run_spec"]
            assert spec["seed"] == entry["seed"]
            assert spec["dependencies"] == {"init_sha256": plan["source_identity"]["sft_sha256"],
                                             "reward_sha256": plan["source_identity"]["rm_sha256"]}
            entry["training_code_sha256"] = spec["code_sha256"]
    plan["origin"] = {"basis": "New run: primary best.pt were selected on validation; no test metrics were used."}
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import torch
    from recsys.training import read_checkpoint

    template = json.loads((ROOT / "configs/final_selection.json").read_text())
    plan = freeze(args.base, template, lambda path: read_checkpoint(path, torch.device("cpu")))
    if args.output.exists() and json.loads(args.output.read_text()) != plan:
        raise ValueError("Фиксация уже существует и отличается: не меняйте ее после оценки теста")
    write_json(args.output, plan)
    print("Состояния зафиксированы до тестовой оценки:", args.output)


if __name__ == "__main__":
    main()
