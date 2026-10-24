"""Persistent wall-clock limits for a Colab experiment, including restarts."""

import math
import time
from contextlib import contextmanager
from pathlib import Path


def new_deadline(minutes, now=None):
    if not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("Positive finite budget is required")
    return (time.time() if now is None else now) + minutes * 60


def remaining_seconds(deadline, now=None):
    return max(0., deadline - (time.time() if now is None else now))


def may_train(deadline, reserve_minutes, stage_elapsed, stage_minutes, now=None):
    return (remaining_seconds(deadline, now) > reserve_minutes * 60
            and stage_elapsed < stage_minutes * 60)


def eligible_checkpoint(stage, metrics, best_metrics, reference_ndcg, fraction=.95):
    if stage == "sft":
        return metrics["ndcg@10"] > best_metrics["ndcg@10"]
    if stage != "rl":
        raise ValueError("Extended training supports sft and rl")
    return (metrics["ndcg@10"] >= fraction * reference_ndcg
            and metrics["rm_reward@10"] > best_metrics["rm_reward@10"] + 1e-6)


def evaluation_due(step, interval):
    return step in {1, 100, 250, 500, 1000} or step % interval == 0


def rl_study_evaluation_due(step, interval):
    return step in {0, 25, 50, 100} or step % interval == 0


def rl_study_utility(metrics, repeat_penalty):
    return metrics["rm_reward@10"] - repeat_penalty * metrics["seen_fraction"]


def rl_study_candidate(metrics, best, reference, fraction, repeat_penalty):
    return (metrics["ndcg@10"] >= fraction * reference["ndcg@10"]
            and rl_study_utility(metrics, repeat_penalty) > rl_study_utility(best, repeat_penalty) + 1e-6)


@contextmanager
def exclusive_run(directory):
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This experiment is already running; do not start a second copy") from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
