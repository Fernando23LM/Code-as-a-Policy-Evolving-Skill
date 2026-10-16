import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from biolab import TubeRackEnv  # noqa: E402
from scripted_policy import policy  # noqa: E402


def test_scene_is_stable_and_unsolved_at_reset():
    env = TubeRackEnv(seed=0)
    before = {k: v["bottom"].copy() for k, v in env.perceive()["tubes"].items()}
    env.step(2.0)
    for name, tube in env.perceive()["tubes"].items():
        assert np.linalg.norm(tube["bottom"] - before[name]) < 1e-3
        assert tube["axis"][2] > 0.999
    result = env.evaluate()
    assert not result["success"] and not result["checks"]["in_target_cell"]


def test_reset_is_deterministic():
    a, b = TubeRackEnv(seed=3), TubeRackEnv(seed=3)
    assert a.task == b.task and a.layout == b.layout


@pytest.mark.parametrize("seed", range(5))
def test_scripted_policy_solves(seed):
    env = TubeRackEnv(seed=seed)
    assert policy(env)
    assert env.evaluate()["success"]


def test_unreachable_target_is_refused():
    env = TubeRackEnv(seed=0)
    start = env.ee_pose("right")["pos"]
    assert not env.move_to("right", [0.9, 0.0, 0.1])
    assert np.linalg.norm(env.ee_pose("right")["pos"] - start) < 1e-3
