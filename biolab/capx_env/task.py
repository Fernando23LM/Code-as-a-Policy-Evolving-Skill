"""CaP-X task: move one vial from the source rack to a given cell of the destination rack."""
from __future__ import annotations

from capx.envs.tasks.base import CodeExecutionEnvBase

from ..api import REFERENCE_POLICY

PROMPT = """
You are controlling two I2RT YAM robot arms on a lab bench with the API described below.
Goal: move one sample vial from the source rack into one cell of the destination rack. Call get_task() to learn which vial and which cell.
The episode succeeds when that vial stands upright and seated in the target cell, the gripper has let go of it, and the other vials are still in their cells.
You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
The functions (APIs) below are already imported to the environment. If you want to use numpy, you need to import it explicitly.
"""



class TubeRackCodeEnv(CodeExecutionEnvBase):
    prompt = PROMPT
    oracle_code = REFERENCE_POLICY
