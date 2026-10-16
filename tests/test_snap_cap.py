import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biolab import SnapCapEnv  # noqa: E402
from biolab.evolve.runner import run_episode  # noqa: E402
from biolab.scene import LATCH_TORQUE  # noqa: E402
from biolab.tasks import TASKS  # noqa: E402

SPEC = TASKS["snap_cap"]


def test_lids_start_shut_and_unsolved():
    env = SnapCapEnv(seed=0)
    env.step(2.0)
    assert all(abs(env.cap_angle(name)) < 1e-3 for name in env.layout.tubes)
    result = env.evaluate()
    assert not result["success"] and not result["checks"]["cap_open"] and result["checks"]["others_shut"]


def test_reset_is_deterministic():
    a, b = SnapCapEnv(seed=3), SnapCapEnv(seed=3)
    assert a.task == b.task and a.layout == b.layout


@pytest.mark.parametrize("torque, opens", [(0.9 * LATCH_TORQUE, False), (1.15 * LATCH_TORQUE, True)])
def test_latch_holds_until_its_torque_is_exceeded(torque, opens):
    env = SnapCapEnv(seed=0)
    tube = env.task.tube
    dof = env.model.joint(f"{tube}_hinge").dofadr[0]
    env.data.qfrc_applied[dof] = torque
    env.step(1.0)
    env.data.qfrc_applied[dof] = 0
    env.step(1.0)
    assert (env.cap_angle(tube) > np.pi / 2) == opens


@pytest.mark.parametrize("seed", range(5))
def test_reference_policy_solves(seed):
    result = run_episode(SPEC.env(seed=0), seed, "", SPEC.reference, SPEC.api)
    assert result["error"] is None and result["success"], result


def test_pulling_straight_up_is_not_enough():
    policy = """
import numpy as np
cap = get_cap(get_task()["tube"])
yaw = (np.arctan2(cap["front"][1], cap["front"][0]) + np.pi / 2) % np.pi - np.pi / 2
set_gripper("right", 0.05)
move_to("right", cap["center"] + [0, 0, 0.05], yaw)
move_to("right", cap["center"], yaw, speed=0.06)
set_gripper("right", 0.0)
move_to("right", cap["center"] + [0, 0, 0.04], yaw, speed=0.05)
set_gripper("right", 0.06)
"""
    result = run_episode(SPEC.env(seed=0), 1, "", policy, SPEC.api)
    assert not result["success"] and not result["checks"]["cap_open"]
