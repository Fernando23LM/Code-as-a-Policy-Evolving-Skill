import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biolab import WeighEnv  # noqa: E402
from biolab.evolve.runner import run_episode  # noqa: E402
from biolab.tasks import TASKS  # noqa: E402

SPEC = TASKS["weigh_salt"]


def _grains(env):
    m = env.model
    return [b for b in range(m.nbody) if "grain" in m.body(b).name]


def test_salt_stays_in_the_vial_and_balance_reads_zero():
    env = WeighEnv(seed=0)
    env.step(2.0)
    vial = env.data.body(env.VIAL)
    for b in _grains(env):
        offset = env.data.xpos[b] - vial.xpos
        assert np.hypot(offset[0], offset[1]) < 0.014 and 0.058 < offset[2] < 0.08
    assert abs(env.balance_reading()) < 0.01
    result = env.evaluate()
    assert not result["success"] and result["checks"]["pan_clear"] and result["checks"]["at_rest"]


def test_reset_is_deterministic():
    a, b = WeighEnv(seed=3), WeighEnv(seed=3)
    assert a.task == b.task and a.layout == b.layout


def test_balance_reads_the_mass_dropped_on_the_boat():
    env = WeighEnv(seed=0)
    m, d = env.model, env.data
    boat = env.perceive()["balance"]["boat_center"]
    dropped = _grains(env)[:20]
    for i, b in enumerate(dropped):
        joint = m.body_jntadr[b]
        d.qpos[m.jnt_qposadr[joint]:m.jnt_qposadr[joint] + 3] = [boat[0] + (i % 5 - 2) * 0.006,
                                                                 boat[1] + (i // 5 - 1.5) * 0.006, boat[2] + 0.02]
        d.qvel[m.jnt_dofadr[joint]:m.jnt_dofadr[joint] + 6] = 0
    env.step(2.0)
    assert abs(env.balance_reading() - 1000 * sum(m.body_mass[b] for b in dropped)) < 0.02


def test_each_arm_reaches_only_its_side_and_both_reach_the_handover_rack():
    env = WeighEnv(seed=0)
    obs = env.perceive()
    stock = obs["racks"]["rack_stock"]["cells"][(0, 0)] + [0, 0, 0.06]
    handover = obs["racks"]["rack_handover"]["cells"][(0, 0)] + [0, 0, 0.06]
    boat = obs["balance"]["boat_center"] + [0, 0, 0.08]
    assert env.move_to("left", handover) and env.move_to("left", stock) and not env.move_to("left", boat)
    assert env.move_to("right", handover) and env.move_to("right", boat) and not env.move_to("right", stock)


def test_reference_policy_solves():
    result = run_episode(SPEC.env(seed=0), 0, "", SPEC.reference, SPEC.api)
    assert result["error"] is None and result["success"], result


def test_sub_tasks_share_the_full_task_layouts():
    full, handover, pour = (TASKS[name].env(seed=5) for name in ("weigh_salt", "weigh_handover", "weigh_pour"))
    assert full.layout.balance == handover.layout.balance == pour.layout.balance
    assert full.task.target_g == pour.task.target_g
    start = pour.data.body(pour.VIAL).xpos[:2]
    assert min(np.linalg.norm(start - cell[:2]) for cell in pour.perceive()["racks"]["rack_handover"]["cells"].values()) < 1e-3
    assert not handover.evaluate()["success"] and not pour.evaluate()["success"]


def test_fixed_modules_are_loaded_apart_and_callable():
    spec = TASKS["weigh_handover"]
    modules = {"a": "def name():\n    return helper()\ndef helper():\n    return 'a'\n",
               "b": "def name():\n    return helper()\ndef helper():\n    return 'b'\n"}
    result = run_episode(spec.env(seed=0), 0, "", "print(a.name(), b.name(), get_task()['instruction'][:4])\n",
                         spec.api, None, modules)
    assert result["error"] is None and result["stdout"].startswith("a b Brin")


def test_sub_task_reference_policies_solve():
    for name in ("weigh_handover", "weigh_pour"):
        spec = TASKS[name]
        result = run_episode(spec.env(seed=0), 0, "", spec.reference, spec.api)
        assert result["error"] is None and result["success"], (name, result)
