import pytest

from recsys.budget import eligible_checkpoint, evaluation_due, exclusive_run, may_train, new_deadline, remaining_seconds


def test_deadline_does_not_restart_when_resuming():
    deadline = new_deadline(230, now=1000)
    assert remaining_seconds(deadline, now=1000 + 60 * 30) == 60 * 200
    assert remaining_seconds(deadline, now=deadline + 1) == 0
    with pytest.raises(ValueError):
        new_deadline(float("inf"))


def test_global_reserve_and_stage_limit_stop_training():
    assert may_train(10000, 20, stage_elapsed=100, stage_minutes=50, now=1000)
    assert not may_train(10000, 20, stage_elapsed=100, stage_minutes=50, now=9000)
    assert not may_train(10000, 20, stage_elapsed=3000, stage_minutes=50, now=1000)


def test_rl_selection_optimizes_reward_under_relevance_constraint():
    current = {"ndcg@10": .1, "rm_reward@10": .5}
    # A small accuracy decrease can be accepted if reward improves.
    assert eligible_checkpoint("rl", {"ndcg@10": .096, "rm_reward@10": .6}, current, .1)
    assert not eligible_checkpoint("rl", {"ndcg@10": .09, "rm_reward@10": .9}, current, .1)
    assert not eligible_checkpoint("rl", {"ndcg@10": .11, "rm_reward@10": .49}, current, .1)
    assert not eligible_checkpoint("rl", current, current, .1)


def test_sft_selection_keeps_best_accuracy():
    assert eligible_checkpoint("sft", {"ndcg@10": .11}, {"ndcg@10": .1}, .1)
    assert not eligible_checkpoint("sft", {"ndcg@10": .09}, {"ndcg@10": .1}, .1)


def test_early_snapshots_and_long_run_schedule():
    assert all(evaluation_due(step, 2000) for step in [1, 100, 250, 500, 1000, 2000, 4000])
    assert not evaluation_due(3000, 2000)


def test_second_process_cannot_share_experiment_directory(tmp_path):
    with exclusive_run(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with exclusive_run(tmp_path):
                pass
    with exclusive_run(tmp_path):
        pass
