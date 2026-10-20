"""The tasks a model can be set: environment, API, prompt text and a hand-written reference solution."""
from __future__ import annotations

from dataclasses import dataclass

from .api import (REFERENCE_POLICY, SNAP_CAP_REFERENCE_POLICY, WEIGH_HANDOVER_REFERENCE_POLICY,
                  WEIGH_POUR_REFERENCE_POLICY, WEIGH_REFERENCE_POLICY, SnapCapApi, TubeRackApi, WeighApi, WeighApiPlain,
                  WeighHandoverApi, WeighPourApi)
from .env import SnapCapEnv, TubeRackEnv, WeighEnv, WeighEnvPlain, WeighHandoverEnv, WeighPourEnv


@dataclass(frozen=True)
class TaskSpec:
    env: type
    api: type
    prompt: str
    reference: str
    camera: str      # the view used for videos


TASKS = {
    "tube_rack": TaskSpec(TubeRackEnv, TubeRackApi, """\
You control two I2RT YAM robot arms on a lab bench through the Python API below.
Task: move one sample vial from the source rack into one cell of the destination rack. Call get_task() to learn which vial and which cell; they differ in every episode, and so do the rack positions and angles.
An episode succeeds when that vial stands upright and seated in the target cell, the gripper has let go of it, and the other vial is still in its cell.""", REFERENCE_POLICY, "front"),
    "snap_cap": TaskSpec(SnapCapEnv, SnapCapApi, """\
You control two I2RT YAM robot arms on a lab bench through the Python API below.
Task: open the snap cap of one of two tubes standing in a rack. Call get_task() to learn which tube; it differs in every episode, and so do the rack position and the way each tube is turned.
An episode succeeds when that tube's lid is open by at least 90 degrees and at rest, the gripper has let go of it, and the other tube's lid is still shut.""", SNAP_CAP_REFERENCE_POLICY, "closeup"),
    "weigh_salt": TaskSpec(WeighEnv, WeighApi, """\
You control two I2RT YAM robot arms on a lab bench through the Python API below.
Task: weigh out a target mass of salt. An open vial of salt stands in a rack on the left arm's side of the bench; a balance with a weigh boat on its pan stands on the right arm's side. Neither arm reaches both, so the vial has to pass from one arm to the other. Call get_task() to learn the target mass; it differs in every episode, and so do the positions of the racks and the balance.
An episode succeeds when the balance reads within the tolerance of the target, the reading is steady, and nothing but salt is on the pan and the boat: no gripper and no vial touching them.""", WEIGH_REFERENCE_POLICY, "bench"),
}
# Control for the weighing task: no grip report in the feedback, no sentence about abrupt turns in the API.
TASKS["weigh_salt_plain"] = TaskSpec(WeighEnvPlain, WeighApiPlain, TASKS["weigh_salt"].prompt, WEIGH_REFERENCE_POLICY,
                                     "bench")

# The weighing task in two halves, to be learned separately and then combined (see --modules).
_BENCH = "You control two I2RT YAM robot arms on a lab bench through the Python API below.\n"
TASKS["weigh_handover"] = TaskSpec(WeighHandoverEnv, WeighHandoverApi, _BENCH + """\
Task: bring a vial of salt to where the other arm can take it. An open vial of salt stands in a rack on the left arm's side of the bench, where the right arm cannot reach; a small rack between the arms, rack_handover, is the one place both reach. The positions of the racks differ in every episode.
An episode succeeds when the vial stands upright in a cell of rack_handover, no gripper is touching it, and no salt has spilled.""", WEIGH_HANDOVER_REFERENCE_POLICY, "bench")
TASKS["weigh_pour"] = TaskSpec(WeighPourEnv, WeighPourApi, _BENCH + """\
Task: weigh out a target mass of salt. An open vial of salt stands in the small rack between the arms; a balance with a weigh boat on its pan stands on the right arm's side. Call get_task() to learn the target mass; it differs in every episode, and so do the positions of the rack and the balance.
An episode succeeds when the balance reads within the tolerance of the target, the reading is steady, and nothing but salt is on the pan and the boat: no gripper and no vial touching them.""", WEIGH_POUR_REFERENCE_POLICY, "bench")

# Protein assay (biolab.protein): pipetting, a dilution series and a plate reader, right arm only.
from .protein import PROTEIN_PROMPT, PROTEIN_REFERENCE_POLICY, ProteinApi, ProteinEnv  # noqa: E402

TASKS["protein_assay"] = TaskSpec(ProteinEnv, ProteinApi, PROTEIN_PROMPT, PROTEIN_REFERENCE_POLICY, "bench")

# Cell culture (biolab.culture): pipetting plus a cell growth model over a simulated week.
from .culture import CULTURE_PROMPT, CULTURE_REFERENCE_POLICY, CultureApi, CultureEnv  # noqa: E402

TASKS["cell_culture"] = TaskSpec(CultureEnv, CultureApi, CULTURE_PROMPT, CULTURE_REFERENCE_POLICY, "bench")

# Drug dose-response (biolab.drug): seed cells, treat with a dilution series, read, report the IC50.
from .drug import DRUG_PROMPT, DRUG_REFERENCE_POLICY, DrugApi, DrugEnv  # noqa: E402

TASKS["drug_response"] = TaskSpec(DrugEnv, DrugApi, DRUG_PROMPT, DRUG_REFERENCE_POLICY, "bench")

# Realistic variants: every task again, with the sensing, arm, pipetting and physics errors of biolab.realism.
import functools  # noqa: E402

from .realism import Realism  # noqa: E402

REALITY_NOTE = """
This bench behaves like real hardware. Positions reported by the cameras are off by a millimetre or two (a fixed error for the session plus jitter on every reading), the arm lands about a millimetre away from where it is sent and reports its own position in its own frame, pipetted volumes are off by about 1 percent, readings are noisy, and friction, masses and motor strength differ a little between sessions."""

for _name, _spec in list(TASKS.items()):
    TASKS[f"{_name}_real"] = TaskSpec(functools.partial(_spec.env, realism=Realism()), _spec.api,
                                      _spec.prompt + REALITY_NOTE, _spec.reference, _spec.camera)
