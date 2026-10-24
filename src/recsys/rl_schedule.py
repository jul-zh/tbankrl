"""CPU-only phase rules for equal-update CE/RL schedule comparisons."""


def phase_settings(settings, completed_updates):
    mode = settings.get("training_schedule")
    if mode is None:
        return settings, 1
    modes = {"ce_then_rl": (0., 1.), "rl_then_ce": (1., 0.), "rl_then_rl": (1., 1.), "ce_then_ce": (0., 0.)}
    if mode not in modes:
        raise ValueError("Unknown training schedule")
    boundary = settings.get("switch_step")
    if not isinstance(boundary, int) or isinstance(boundary, bool) or boundary < 1 or completed_updates < 0:
        raise ValueError("Schedule boundary and completed updates must be valid")
    if settings["ce_weight"] <= 0 or settings["horizon"] != 1 or settings["group_size"] != 1:
        raise ValueError("CE/RL schedule requires positive CE weight, H=1, group=1")
    if settings.get("objective_mode", "sampled") != "sampled":
        raise ValueError("Schedule study uses sampled REINFORCE")
    phase = 1 if completed_updates < boundary else 2
    return {**settings, "reinforce_weight": modes[mode][phase - 1]}, phase


def reset_due(settings, completed_updates, saved_phase):
    _, phase = phase_settings(settings, completed_updates)
    return phase > saved_phase


def retain_snapshot(settings, step):
    interval = settings.get("snapshot_every", 0)
    return not interval or step in {0, settings["max_steps"], settings.get("switch_step")} or step % interval == 0


def rl_updates(settings, completed_updates):
    mode = settings.get("training_schedule")
    if mode is None:
        return completed_updates if settings.get("reinforce_weight", 1.) else 0
    phase_settings(settings, completed_updates)
    if mode == "ce_then_rl":
        return max(0, completed_updates - settings["switch_step"])
    if mode == "rl_then_ce":
        return min(completed_updates, settings["switch_step"])
    if mode == "rl_then_rl":
        return completed_updates
    return 0
