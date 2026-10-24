"""CPU checks for frequent early evaluations and reward/relevance selection."""

from recsys.budget import rl_study_candidate, rl_study_evaluation_due, rl_study_utility


def test_modified_reward_selection_preserves_relevance_and_changes_utility():
    reference = {"ndcg@10": .1, "rm_reward@10": .6, "seen_fraction": .2}
    candidate = {"ndcg@10": .096, "rm_reward@10": .59, "seen_fraction": .05}
    assert rl_study_utility(candidate, .2) > rl_study_utility(reference, .2)
    assert rl_study_candidate(candidate, reference, reference, .95, .2)
    assert not rl_study_candidate(candidate, reference, reference, .95, 0.)
    assert not rl_study_candidate({**candidate, "ndcg@10": .09}, reference, reference, .95, .2)
    assert not rl_study_candidate(reference, reference, reference, .95, .2)
    assert all(rl_study_evaluation_due(step, 200) for step in [0, 25, 50, 100, 200, 400, 2000])
