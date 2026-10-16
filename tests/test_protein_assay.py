import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biolab.evolve.runner import run_episode  # noqa: E402
from biolab.liquid import Pipette, Well  # noqa: E402
from biolab.tasks import TASKS  # noqa: E402

SPEC = TASKS["protein_assay"]


def test_pipette_mixes_draws_air_and_spills():
    a, b = Well("a", np.zeros(2), 0.0, 0.004, 0.01), Well("b", np.array([0.1, 0.0]), 0.0, 0.004, 0.01)
    a.add(200, 1.0)
    pipette = Pipette()
    above = np.array([0.0, 0.0, a.surface + 0.002])
    assert pipette.aspirate([a, b], above, 50)["volume_ul"] == 0            # above the surface: air
    assert pipette.aspirate([a, b], np.array([0.0, 0.0, 0.0005]), 50)["volume_ul"] == 50
    b.add(50, 0.0)
    assert pipette.dispense([a, b], np.array([0.1, 0.0, 0.003]), 50)["into"] == "b"
    assert abs(b.conc - 0.5) < 1e-9 and abs(b.volume - 100) < 1e-9
    pipette.aspirate([a, b], np.array([0.0, 0.0, 0.0005]), 30)
    assert pipette.dispense([a, b], np.array([0.5, 0.5, 0.05]), 30)["into"] is None and pipette.spilled == 30


def test_starts_holding_the_pipette_and_unsolved():
    env = SPEC.env(seed=0)
    assert env.holding_pipette()
    env.step(2.0)
    assert env.holding_pipette()
    result = env.evaluate()
    assert not result["success"] and result["progress"] < 0.1


def test_reset_is_deterministic():
    a, b = SPEC.env(seed=4), SPEC.env(seed=4)
    assert a.task == b.task and a.layout == b.layout


def test_reference_policy_solves():
    for seed in (0, 1, 2):
        result = run_episode(SPEC.env(seed=0), seed, "", SPEC.reference, SPEC.api)
        assert result["error"] is None and result["success"], result
