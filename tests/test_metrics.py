import numpy as np
import pytest

from recsys.baseline import regression_metrics
from recsys.metrics import behavior, paired_bootstrap, per_user_ranking, topk


def test_ranking_hand_computed():
    recs = np.array([[3, 4, 5], [5, 4, 3], [3, 5, 4]])
    values = per_user_ranking(recs, np.array([3, 4, 9]), (1, 3))
    np.testing.assert_allclose(values["recall@1"], [1, 0, 0])
    np.testing.assert_allclose(values["recall@3"], [1, 1, 0])
    np.testing.assert_allclose(values["ndcg@3"], [1, 1 / np.log2(3), 0])


def test_special_tokens_excluded_even_with_huge_scores():
    scores = np.array([[100, 200, 300, .5, 2., 2.]])
    assert topk(scores, 20).tolist() == [[4, 5, 3]]


def test_behavior_detects_collapse_and_seen_repeats():
    recs = np.array([[3, 3], [3, 3]])
    values = behavior(recs, np.array([[3, 0], [4, 0]]), np.array([0, 0, 0, 10, 5]))
    assert values["coverage"] == .5
    assert values["within_list_repeat_fraction"] == .5
    assert values["seen_fraction"] == .5
    assert values["top_1pct_popular_share"] == 1.


def test_paired_bootstrap_constant_difference():
    result = paired_bootstrap([.1, .3, .5], [.2, .4, .6], repeats=100)
    np.testing.assert_allclose(result["ci95"], [.1, .1])
    assert result["mean_delta"] == pytest.approx(.1)


def test_reward_metrics_ignore_missing_and_handle_constant_prediction():
    values = regression_metrics(np.array([.5, .5, .2]), np.array([0., 1., np.nan]))
    assert values["n"] == 2
    assert values["mae"] == .5
    assert values["rmse"] == .5
    assert values["spearman"] is None
    assert regression_metrics(np.array([.2]), np.array([np.nan]))["mae"] is None
