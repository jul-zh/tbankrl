"""Schedule boundary, reset-once behavior and sparse checkpoint retention."""

import pytest

from recsys.rl_schedule import phase_settings, reset_due, retain_snapshot, rl_updates


def settings(mode):
    return {"training_schedule": mode, "switch_step": 2500, "max_steps": 5000,
            "ce_weight": .3, "horizon": 1, "group_size": 1, "snapshot_every": 500}


@pytest.mark.parametrize("mode,first,second", [("ce_then_rl", 0., 1.), ("rl_then_ce", 1., 0.), ("rl_then_rl", 1., 1.), ("ce_then_ce", 0., 0.)])
def test_phase_switch_after_2500_complete_updates(mode, first, second):
    config = settings(mode)
    assert phase_settings(config, 0)[0]["reinforce_weight"] == first
    assert phase_settings(config, 2499)[0]["reinforce_weight"] == first
    effective, phase = phase_settings(config, 2500)
    assert phase == 2 and effective["reinforce_weight"] == second
    assert phase_settings(config, 4999)[0]["reinforce_weight"] == second
    assert "reinforce_weight" not in config
    assert not reset_due(config, 2499, 1)
    assert reset_due(config, 2500, 1)
    assert not reset_due(config, 2500, 2)
    assert not reset_due(config, 4999, 2)


def test_snapshot_policy_preserves_boundaries_and_final():
    config = settings("ce_then_rl")
    assert all(retain_snapshot(config, step) for step in [0, 500, 2500, 5000])
    assert not retain_snapshot(config, 100)
    assert retain_snapshot({**config, "max_steps": 4999}, 4999)
    assert retain_snapshot({"max_steps": 10}, 3)


def test_invalid_schedule_cannot_silently_train_different_objective():
    config = settings("ce_then_rl")
    for patch in [{"training_schedule": "unknown"}, {"switch_step": 0}, {"switch_step": True},
                  {"horizon": 5}, {"group_size": 4}, {"ce_weight": 0}, {"objective_mode": "full_catalog"}]:
        with pytest.raises(ValueError):
            phase_settings({**config, **patch}, 0)
    plain = {"name": "old_run"}
    assert phase_settings(plain, 2500) == (plain, 1)
    assert not reset_due(plain, 2500, 1)


def test_count_rl_updates_without_calling_ce_only_checkpoint_rl():
    assert rl_updates(settings("ce_then_rl"), 2500) == 0
    assert rl_updates(settings("ce_then_rl"), 2501) == 1
    assert rl_updates(settings("ce_then_rl"), 5000) == 2500
    assert rl_updates(settings("rl_then_ce"), 100) == 100
    assert rl_updates(settings("rl_then_ce"), 5000) == 2500
    assert rl_updates(settings("rl_then_rl"), 5000) == 5000
    assert rl_updates(settings("ce_then_ce"), 5000) == 0
