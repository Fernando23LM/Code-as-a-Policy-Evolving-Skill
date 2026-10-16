import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biolab.evolve.runner import run_episode  # noqa: E402
from biolab.tasks import TASKS  # noqa: E402

SPEC = TASKS["cell_culture"]


def test_cells_grow_then_starve_without_fresh_medium():
    env = SPEC.env(seed=0)
    start = env.attached["A"]
    env.incubate(24)
    assert env.attached["A"] > 1.6 * start and env.starved["A"] == 0
    for _ in range(4):
        env.incubate(24)
    assert env.starved["A"] > 0 and env.viability() < 1.0


def test_trypsin_lifts_cells_and_medium_lets_them_settle():
    env = SPEC.env(seed=0)
    dish = env.wells["A"]
    cells = env.attached["A"]
    dish.take(dish.volume)
    dish.add(300, 1.0)                           # trypsin
    env.incubate(0.1)
    assert env.attached["A"] < 0.1 * cells and dish.extra["cells"] > 0.8 * cells
    dish.add(900, 0.0, {"nutrient": 900.0})      # dilute it
    env.incubate(2.0)
    assert env.attached["A"] > 0.7 * cells


def test_reset_is_deterministic():
    a, b = SPEC.env(seed=4), SPEC.env(seed=4)
    assert a.task == b.task and a.layout == b.layout


def test_reference_policy_solves():
    result = run_episode(SPEC.env(seed=0), 0, "", SPEC.reference, SPEC.api)
    assert result["error"] is None and result["success"], result
