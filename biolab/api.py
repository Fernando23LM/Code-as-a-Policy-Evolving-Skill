"""Functions exposed to code-writing models. The docstrings are all a model sees.

Plain Python, no CaP-X imports, so the same API serves CaP-X (biolab.capx_env)
and the library loop (biolab.evolve).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .env import SnapCapEnv, TubeRackEnv, WeighEnv


class TubeRackApi:
    """Control of the two YAM arms plus ground-truth scene state."""

    def __init__(self, sim: TubeRackEnv) -> None:
        self.sim = sim

    def functions(self) -> dict[str, Any]:
        return {
            "get_task": self.get_task,
            "get_tube_pose": self.get_tube_pose,
            "get_rack": self.get_rack,
            "get_gripper_pose": self.get_gripper_pose,
            "move_to": self.move_to,
            "set_gripper": self.set_gripper,
            "go_home": self.go_home,
        }

    def get_task(self) -> dict:
        """Get the goal of this episode.

        Returns:
            dict with keys:
              "instruction": str, the goal in words.
              "tube": str, name of the vial to move, e.g. "tube_1".
              "dest_cell": (ix, iy) tuple of ints, the target cell of "rack_dest".
        """
        task = self.sim.task
        return {"instruction": task.instruction, "tube": task.tube, "dest_cell": tuple(task.dest_cell)}

    def get_tube_pose(self, tube: str) -> tuple[np.ndarray, np.ndarray]:
        """Get where a vial is. Vials are capped cylinders, 28 mm diameter and 75 mm tall.

        Args:
            tube: vial name, "tube_0" or "tube_1".

        Returns:
            bottom: (3,) XYZ in metres of the centre of the vial's bottom face, world frame.
            top: (3,) XYZ in metres of the centre of the vial's cap.
        """
        state = self.sim.observe()["tubes"][tube]
        return state["bottom"].copy(), state["top"].copy()

    def get_rack(self, rack: str) -> dict:
        """Get a rack's angle and the positions of its cells.

        Racks are fixed to the table, 35 mm tall, with a 4 x 2 grid of square cells
        (32 mm opening, 37 mm pitch). Cell (ix, iy) has ix in 0..3 along the rack's
        long side and iy in 0..1 along its short side.

        Args:
            rack: "rack_source" (holds the vials at the start) or "rack_dest" (empty).

        Returns:
            dict with keys:
              "yaw": float, rotation of the rack's long side about world Z, radians.
              "cells": dict mapping (ix, iy) to (3,) XYZ in metres of the point where
                a seated vial's bottom centre rests.
        """
        state = self.sim.observe()["racks"][rack]
        return {"yaw": float(state["yaw"]), "cells": {c: p.copy() for c, p in state["cells"].items()}}

    def get_gripper_pose(self, arm: str) -> dict:
        """Get the current gripper state of one arm.

        Args:
            arm: "left" or "right".

        Returns:
            dict with keys "pos" ((3,) XYZ in metres of the pinch point between the
            finger pads), "yaw" (radians) and "opening" (metres between the pads).
        """
        return self.sim.reported_ee_pose(arm)

    def move_to(self, arm: str, position: np.ndarray, yaw: float = 0.0, tilt: float = 0.0,
                speed: float = 0.15) -> bool:
        """Move the pinch point of an arm's gripper in a straight line to a position.

        World frame: metres, Z up, table top at z = 0. The right arm's base is at
        (-0.20, -0.28, 0) and the left arm's at (-0.20, 0.28, 0), both facing +X.
        Only the right arm can reach the racks.

        Args:
            arm: "left" or "right".
            position: (3,) XYZ target of the pinch point in metres.
            yaw: rotation of the gripper about world Z in radians. At yaw = 0 the
              fingers close along world Y. To close the fingers across a rack's
              short side, use the rack's yaw.
            tilt: 0 points the gripper straight down; positive leans it forward.
              Straight down is always reachable over the racks below z = 0.15.
              Between 0.15 and 0.19 it depends on the position, and above that
              it is not reachable, so check the return value.
            speed: metres per second. Use about 0.05 to 0.08 for the last
              centimetres of a grasp or an insertion.

        Returns:
            True if the target was reached within 1 mm. False if the target is
            unreachable, in which case the arm DOES NOT MOVE AT ALL, or if the
            arm was blocked on the way, for example by a vial resting on a rack
            wall. Continuing after a False usually drags or drops what is held.
        """
        return bool(self.sim.command(arm, np.asarray(position, dtype=float), yaw, tilt, speed))

    def set_gripper(self, arm: str, opening: float) -> None:
        """Open or close a gripper and wait for it to settle.

        The fingers do not fit between vials in touching cells. The two vials
        never start in touching cells, so at the source rack the cells around
        the vial to grasp are free.

        Args:
            arm: "left" or "right".
            opening: commanded distance between the finger pads in metres, from
              0.0 to 0.082. The fingers are position-controlled springs: they
              stop on whatever they hold and squeeze with a force proportional
              to how far the command is below the object (about 0.5 N per mm,
              40 N at most). Command 0.0 for a firm grip. A held 28 mm vial
              reads as an opening of about 0.025, so commanding 0.025 applies
              almost no force.
        """
        self.sim.set_gripper(arm, float(opening))

    def go_home(self, arm: str) -> None:
        """Move an arm back to its resting pose, clear of the racks.

        Args:
            arm: "left" or "right".
        """
        self.sim.home(arm)


REFERENCE_POLICY = """
task = get_task()
bottom, top = get_tube_pose(task["tube"])
source, dest = get_rack("rack_source"), get_rack("rack_dest")

travel_z = 0.152                      # carried vial clears the standing ones
grasp = top.copy()
grasp[2] -= 0.018                     # pinch just below the cap
place = dest["cells"][task["dest_cell"]].copy()
place[2] += grasp[2] - bottom[2] + 0.004   # let go 4 mm above the rack base

set_gripper("right", 0.05)
ok = move_to("right", [grasp[0], grasp[1], travel_z], source["yaw"])
ok = ok and move_to("right", grasp, source["yaw"], speed=0.08)
set_gripper("right", 0.0)
ok = ok and move_to("right", [grasp[0], grasp[1], travel_z], source["yaw"], speed=0.08)
ok = ok and move_to("right", [place[0], place[1], travel_z], dest["yaw"])
ok = ok and move_to("right", place, dest["yaw"], speed=0.05)
set_gripper("right", 0.05)
ok = ok and move_to("right", [place[0], place[1], travel_z], dest["yaw"], speed=0.08)
go_home("right")
print("all moves reached:", ok)
"""


class SnapCapApi(TubeRackApi):
    """Control of the two YAM arms plus ground-truth state of the snap-cap tubes."""

    def __init__(self, sim: SnapCapEnv) -> None:
        super().__init__(sim)

    def functions(self) -> dict[str, Any]:
        return {
            "get_task": self.get_task,
            "get_cap": self.get_cap,
            "get_gripper_pose": self.get_gripper_pose,
            "move_to": self.move_to,
            "set_gripper": self.set_gripper,
            "go_home": self.go_home,
        }

    def get_task(self) -> dict:
        """Get the goal of this episode.

        Returns:
            dict with keys:
              "instruction": str, the goal in words.
              "tube": str, name of the tube whose lid to open, e.g. "tube_1".
        """
        task = self.sim.task
        return {"instruction": task.instruction, "tube": task.tube}

    def get_cap(self, tube: str) -> dict:
        """Get the lid of a tube: where its hinge is and how far it is open.

        Each tube is a cylinder 22 mm in diameter and 75 mm tall with its lid shut,
        clamped upright in the rack: it cannot be lifted out or turned. The rack
        is 35 mm tall. The lid is a flat rectangular plate 30 mm long (hinge to
        front edge), 28 mm wide (along the hinge) and 10 mm thick. The hinge runs
        along the lid's back bottom edge, level with the tube's rim, so opening
        swings the front edge up and back over the hinge. The lid stops at about
        125 degrees.

        A snap latch holds a shut lid down. It gives way once the lid is a few
        degrees open, and overcoming it takes about 3 N at the front edge (more
        nearer the hinge). Past the latch the lid turns freely and stays at the
        angle it is left at.

        The two tubes are turned different ways and are never in touching cells.

        Args:
            tube: tube name, "tube_0" or "tube_1".

        Returns:
            dict with keys:
              "hinge": (3,) XYZ in metres of the middle of the hinge line, world frame.
              "front": (3,) horizontal unit vector pointing from the hinge to the
                front edge of the shut lid.
              "center": (3,) XYZ of the middle of the lid (mid-length, mid-width,
                mid-thickness) as it is now.
              "angle": float, how far the lid is open in radians, 0 when shut.
        """
        state = self.sim.observe()["tubes"][tube]
        return {"hinge": state["hinge"].copy(), "front": state["front"].copy(),
                "center": state["cap_center"].copy(), "angle": float(state["angle"])}

    def set_gripper(self, arm: str, opening: float) -> None:
        """Open or close a gripper and wait for it to settle.

        Each finger is a flat plate 12 mm wide. The fingertips reach 17 mm below
        the pinch point, and the fingers always hang straight down from the
        gripper (or lean with `tilt`); they do not turn with what they hold.

        Args:
            arm: "left" or "right".
            opening: commanded distance between the finger pads in metres, from
              0.0 to 0.082. The fingers are position-controlled springs: they
              stop on whatever they hold and squeeze with a force proportional
              to how far the command is below the object (about 0.5 N per mm,
              40 N at most). Command 0.0 for a firm grip.
        """
        self.sim.set_gripper(arm, float(opening))


SNAP_CAP_REFERENCE_POLICY = """
import numpy as np

task = get_task()
cap = get_cap(task["tube"])
hinge, front, up = cap["hinge"], cap["front"], np.array([0.0, 0.0, 1.0])
# Fingers close along the hinge line, so the lid can turn between the pads. Either way round will do.
yaw = (np.arctan2(front[1], front[0]) + np.pi / 2) % np.pi - np.pi / 2

def lid_middle(angle):
    along, across = np.cos(angle) * front + np.sin(angle) * up, -np.sin(angle) * front + np.cos(angle) * up
    return hinge + 0.015 * along + 0.005 * across

start = lid_middle(0.0)
set_gripper("right", 0.05)
ok = move_to("right", start + [0, 0, 0.05], yaw)
ok = ok and move_to("right", start, yaw, speed=0.06)
set_gripper("right", 0.0)
# Follow the lid's arc to upright. The lid slips in the pads and lags the gripper, so go by the measured angle.
for target in np.radians(np.arange(10, 131, 10)):
    move_to("right", lid_middle(target), yaw, speed=0.05)   # often False: the lid holds the pads off target
    if get_cap(task["tube"])["angle"] > np.radians(80):
        break
set_gripper("right", 0.06)
here = get_gripper_pose("right")["pos"]
move_to("right", [here[0], here[1], hinge[2] + 0.07], yaw, speed=0.05)
# Finish with a push: shut fingers come down in front of the standing lid and press it back to its stop.
set_gripper("right", 0.0)
push = hinge + 0.022 * front + 0.03 * up
move_to("right", push + 0.04 * up, yaw, speed=0.08)
move_to("right", push, yaw, speed=0.05)
for back in np.arange(0.019, -0.012, -0.003):
    move_to("right", hinge + back * front + 0.03 * up, yaw, speed=0.03)
    if get_cap(task["tube"])["angle"] > np.radians(112):
        break
here = get_gripper_pose("right")["pos"]
move_to("right", here + 0.006 * front, yaw, speed=0.05)                # back off the lid before lifting
move_to("right", [here[0], here[1], 0.15], yaw, speed=0.08)            # well clear: go_home sweeps low enough to knock the lid shut
go_home("right")
print("approach ok:", ok, " lid angle:", round(get_cap(task["tube"])["angle"], 2))
"""


class WeighApi(TubeRackApi):
    """Control of the two YAM arms, ground-truth poses and the balance reading."""

    def __init__(self, sim: WeighEnv) -> None:
        super().__init__(sim)

    def functions(self) -> dict[str, Any]:
        return {
            "get_task": self.get_task,
            "get_tube_pose": self.get_tube_pose,
            "get_rack": self.get_rack,
            "get_balance": self.get_balance,
            "get_gripper_pose": self.get_gripper_pose,
            "move_to": self.move_to,
            "set_gripper": self.set_gripper,
            "go_home": self.go_home,
            "wait": self.wait,
        }

    def get_task(self) -> dict:
        """Get the goal of this episode.

        Returns:
            dict with keys:
              "instruction": str, the goal in words.
              "target_g": float, grams of salt the balance should read.
              "tolerance_g": float, how far from the target the final reading may be.
        """
        task = self.sim.task
        return {"instruction": task.instruction, "target_g": task.target_g, "tolerance_g": task.tolerance_g}

    def get_tube_pose(self, tube: str = "salt_vial") -> tuple[np.ndarray, np.ndarray]:
        """Get where the salt vial is.

        The vial is a cylinder 28 mm in diameter and 75 mm tall with no cap. Only
        its top 15 mm is hollow: a cup, 14 mm across at its floor and widening to
        the rim, that starts nearly full with 2.4 g of coarse salt (68 loose
        grains, 2.8 to 3.6 mm across, 0.02 to 0.05 g each). Below the cup the
        vial is solid. The salt stays in while the vial is upright and spills
        when it leans far enough; spilled salt cannot be put back.

        Args:
            tube: "salt_vial".

        Returns:
            bottom: (3,) XYZ in metres of the centre of the vial's bottom face, world frame.
            top: (3,) XYZ in metres of the centre of the vial's open rim.
        """
        state = self.sim.observe()["tubes"][tube]
        return state["bottom"].copy(), state["top"].copy()

    def get_rack(self, rack: str) -> dict:
        """Get a rack's angle and the positions of its cells.

        Racks are fixed to the table and 35 mm tall, with square cells (32 mm
        opening, 37 mm pitch). "rack_stock" (4 x 2 cells) holds the salt vial at
        the start and is within reach of the left arm only. "rack_handover"
        (2 x 1 cells, empty) stands between the arms, within reach of both.

        Args:
            rack: "rack_stock" or "rack_handover".

        Returns:
            dict with keys:
              "yaw": float, rotation of the rack's long side about world Z, radians.
              "cells": dict mapping (ix, iy) to (3,) XYZ in metres of the point where
                a seated vial's bottom centre rests.
        """
        state = self.sim.observe()["racks"][rack]
        return {"yaw": float(state["yaw"]), "cells": {c: p.copy() for c, p in state["cells"].items()}}

    def get_balance(self) -> dict:
        """Read the balance and get where its weigh boat is.

        The balance is within reach of the right arm only. A square weigh boat
        with a 12 mm rim sits on its pan and was tared, so the reading is the
        mass that has landed on the pan since the start. Anything that rests or
        presses on the pan or the boat is weighed too.

        Returns:
            dict with keys:
              "reading_g": float, grams shown on the display. The display averages
                over the last half second and settles about a second after
                something lands.
              "boat_center": (3,) XYZ in metres of the centre of the boat's floor.
              "boat_size": float, inner side length of the boat in metres (0.064).
        """
        state = self.sim.observe()["balance"]
        return {"reading_g": float(state["reading_g"]), "boat_center": state["boat_center"].copy(),
                "boat_size": float(state["boat_size"])}

    def move_to(self, arm: str, position: np.ndarray, yaw: float = 0.0, tilt: float = 0.0,
                speed: float = 0.15) -> bool:
        """Move the pinch point of an arm's gripper in a straight line to a position.

        World frame: metres, Z up, table top at z = 0. The right arm's base is at
        (-0.20, -0.28, 0) and the left arm's at (-0.20, 0.28, 0), both facing +X.
        Each arm reaches only its own side of the table and a narrow strip
        between the arms.

        Args:
            arm: "left" or "right".
            position: (3,) XYZ target of the pinch point in metres.
            yaw: rotation of the gripper about world Z in radians. At yaw = 0 the
              fingers close along world Y. To close the fingers across a rack's
              short side, use the rack's yaw.
            tilt: 0 points the gripper straight down. A positive tilt leans it
              about the line between the finger pads, swinging the fingertips
              away from the arm along the yaw direction and the wrist back
              towards it; whatever the fingers hold leans with them. Straight
              down is reachable below about z = 0.15. Over the balance the right
              arm can lean to about 1.4 rad. The gripper turns to the new yaw
              and tilt at the very start of a move, as fast as the joints allow
              and with no ramp, so a large change in one call jerks whatever
              the fingers hold.
            speed: metres per second. Use about 0.05 to 0.08 for the last
              centimetres of a grasp or an insertion.

        Returns:
            True if the target was reached within 1 mm. False if the target is
            unreachable, in which case the arm DOES NOT MOVE AT ALL, or if the
            arm was blocked on the way.
        """
        return bool(self.sim.command(arm, np.asarray(position, dtype=float), yaw, tilt, speed))

    def set_gripper(self, arm: str, opening: float) -> None:
        """Open or close a gripper and wait for it to settle.

        Each finger is a flat plate 12 mm wide. The fingertips reach 17 mm below
        the pinch point.

        Args:
            arm: "left" or "right".
            opening: commanded distance between the finger pads in metres, from
              0.0 to 0.082. The fingers are position-controlled springs: they
              stop on whatever they hold and squeeze with a force proportional
              to how far the command is below the object (about 0.5 N per mm,
              40 N at most). Command 0.0 for a firm grip.
        """
        self.sim.set_gripper(arm, float(opening))

    def wait(self, seconds: float) -> None:
        """Hold still and let time pass, for example while the balance settles.

        Args:
            seconds: how long to wait, at most 5.
        """
        self.sim.step(float(np.clip(seconds, 0.0, 5.0)))


class WeighHandoverApi(WeighApi):
    """The weighing API for the hand-over sub-task: no balance, no target."""

    def functions(self) -> dict[str, Any]:
        functions = super().functions()
        del functions["get_balance"]
        return functions

    def get_task(self) -> dict:
        """Get the goal of this episode.

        Returns:
            dict with key "instruction": str, the goal in words.
        """
        return {"instruction": self.sim.task.instruction}


_STARTS_IN_STOCK = """"rack_stock" (4 x 2 cells) holds the salt vial at
        the start and is within reach of the left arm only. "rack_handover"
        (2 x 1 cells, empty) stands between the arms, within reach of both."""


class WeighPourApi(WeighApi):
    """The weighing API for the pour sub-task, where the vial starts in the hand-over rack."""

    def get_rack(self, rack: str) -> dict:
        return super().get_rack(rack)

    assert _STARTS_IN_STOCK in WeighApi.get_rack.__doc__
    get_rack.__doc__ = WeighApi.get_rack.__doc__.replace(_STARTS_IN_STOCK, """"rack_stock" (4 x 2 cells, empty) is within
        reach of the left arm only. "rack_handover" (2 x 1 cells) holds the
        salt vial at the start and stands between the arms, within reach of both.""")


JERK_NOTE = """ The gripper turns to the new yaw
              and tilt at the very start of a move, as fast as the joints allow
              and with no ramp, so a large change in one call jerks whatever
              the fingers hold."""


class WeighApiPlain(WeighApi):
    """The same API with the sentence about abrupt turns left out of move_to's description."""

    def move_to(self, arm: str, position: np.ndarray, yaw: float = 0.0, tilt: float = 0.0,
                speed: float = 0.15) -> bool:
        return super().move_to(arm, position, yaw, tilt, speed)

    assert JERK_NOTE in WeighApi.move_to.__doc__
    move_to.__doc__ = WeighApi.move_to.__doc__.replace(JERK_NOTE, "")


_WEIGH_PRELUDE = """
import numpy as np

TRAVEL = 0.152                        # carried vial clears the racks
GRIP = 0.025                          # pads this far below the rim: near the middle, so the vial does not swing when leant

def pick(arm, yaw):
    bottom, top = get_tube_pose("salt_vial")
    grasp = top.copy()
    grasp[2] -= GRIP
    set_gripper(arm, 0.05)
    move_to(arm, [grasp[0], grasp[1], TRAVEL], yaw)
    move_to(arm, grasp, yaw, speed=0.08)
    set_gripper(arm, 0.0)
    move_to(arm, [grasp[0], grasp[1], TRAVEL], yaw, speed=0.06)
    return grasp[2] - bottom[2]       # how high on the vial the pads are

def place(arm, cell, yaw, grip_height):
    spot = cell.copy()
    spot[2] += grip_height + 0.004    # let go 4 mm above the rack base
    move_to(arm, [spot[0], spot[1], TRAVEL], yaw, speed=0.1)
    move_to(arm, spot, yaw, speed=0.05)
    set_gripper(arm, 0.05)
    move_to(arm, [spot[0], spot[1], TRAVEL], yaw, speed=0.08)

"""

_WEIGH_SETUP = """task = get_task()
stock, handover = get_rack("rack_stock"), get_rack("rack_handover")
"""

_WEIGH_HANDOVER = """cell = handover["cells"][(0, 0)]

# The left arm brings the vial to the rack between the arms; the right arm takes it from there.
height = pick("left", stock["yaw"])
place("left", cell, handover["yaw"], height)
go_home("left")
"""

_WEIGH_FIND_CELL = """bottom, _ = get_tube_pose("salt_vial")
cell = min(handover["cells"].values(), key=lambda c: float(np.linalg.norm(c[:2] - bottom[:2])))   # put it back where it stood
"""

_WEIGH_RIGHT_PICK = """height = pick("right", handover["yaw"])

"""

_WEIGH_POUR = """boat = get_balance()["boat_center"]
yaw = float(np.arctan2(boat[1] + 0.28, boat[0] + 0.20))       # pointing away from the right arm's base
forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])

def pour_pose(tilt):
    # The rim is GRIP behind the pinch point along the gripper; keep it over the middle of the boat
    # and the vial's bottom edge well clear of the boat's rim.
    approach = forward * np.sin(tilt) - np.array([0.0, 0.0, np.cos(tilt)])
    spot = np.array([boat[0], boat[1], 0.0]) + GRIP * approach
    spot[2] = 0.065 + (0.075 - GRIP) * np.cos(tilt) + 0.014 * np.sin(tilt)
    return spot

move_to("right", [pour_pose(0.0)[0], pour_pose(0.0)[1], TRAVEL], yaw, speed=0.1)
move_to("right", pour_pose(0.0), yaw, speed=0.08)
tilt = 0.0
for tilt in np.arange(0.1, 0.71, 0.1):                         # nothing spills this early
    move_to("right", pour_pose(tilt), yaw, tilt, speed=0.03)
# Salt leaves in small avalanches, so go quickly to within 0.3 g and then creep, waiting for the display.
target = task["target_g"]
while tilt < 1.40 and get_balance()["reading_g"] < target - 0.3:
    tilt += 0.01
    move_to("right", pour_pose(tilt), yaw, tilt, speed=0.03)
    wait(0.5)
while tilt < 1.40:
    wait(0.8)
    if get_balance()["reading_g"] >= target - 0.06:
        break
    tilt += 0.004
    move_to("right", pour_pose(tilt), yaw, tilt, speed=0.02)
for back in np.arange(tilt - 0.2, -0.01, -0.2).clip(0.0):       # stand the vial up again
    move_to("right", pour_pose(back), yaw, back, speed=0.03)
move_to("right", pour_pose(0.0), yaw, speed=0.03)
move_to("right", [pour_pose(0.0)[0], pour_pose(0.0)[1], TRAVEL], yaw, speed=0.08)
place("right", cell, handover["yaw"], height)
go_home("right")
wait(1.0)
print("target", task["target_g"], "reading", get_balance()["reading_g"], "stopped at tilt", round(float(tilt), 2))
"""

WEIGH_REFERENCE_POLICY = _WEIGH_PRELUDE + _WEIGH_SETUP + _WEIGH_HANDOVER + _WEIGH_RIGHT_PICK + _WEIGH_POUR
WEIGH_HANDOVER_REFERENCE_POLICY = (_WEIGH_PRELUDE + _WEIGH_SETUP + _WEIGH_HANDOVER
                                   + 'print("vial delivered:", get_tube_pose("salt_vial")[0])\n')
WEIGH_POUR_REFERENCE_POLICY = _WEIGH_PRELUDE + _WEIGH_SETUP + _WEIGH_FIND_CELL + _WEIGH_RIGHT_PICK + _WEIGH_POUR
