import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biolab.drug import SEED, DrugTask, grow, true_ic50  # noqa: E402
from biolab.evolve.runner import run_episode  # noqa: E402
from biolab.tasks import TASKS  # noqa: E402

SPEC = TASKS["drug_response"]


def test_drug_slows_growth_more_at_higher_dose():
    task = DrugTask(ic50=0.5, hill=1.2, doubling=24.0)
    untreated, low, high = (grow(SEED, c, task, 48)[0] for c in (0.0, 0.1, 10.0))
    assert untreated > low > high and high < SEED


def test_true_ic50_halves_the_untreated_cells():
    task = DrugTask(ic50=0.5, hill=1.2, doubling=24.0)
    settled = grow(SEED, 0.0, task, 24)[0]
    control = grow(settled, 0.0, task, 48)[0]
    assert abs(grow(settled, true_ic50(task), task, 48)[0] / control - 0.5) < 0.01


def test_reset_is_deterministic_and_unsolved():
    a, b = SPEC.env(seed=4), SPEC.env(seed=4)
    assert a.task == b.task and a.layout == b.layout and a.holding_pipette()
    assert not a.evaluate()["success"]


def test_reference_policy_solves():
    result = run_episode(SPEC.env(seed=0), 0, "", SPEC.reference, SPEC.api)
    assert result["error"] is None and result["success"], result
