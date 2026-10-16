"""Protein assay: make a dilution series of a protein standard, read it on a plate, and measure an unknown.

The right arm holds a pipette from the start. Liquids sit in a reservoir block (stock protein, buffer,
the unknown sample, waste) and are pipetted into a 2 x 4 plate. Reading the plate gives a Bradford-like
absorbance that saturates with concentration, with its curve changing from episode to episode, so the
unknown can only be measured against standards made in the same episode. Liquids are bookkeeping
(biolab.liquid), not fluid simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from .api import TubeRackApi
from .env import TubeRackEnv
from .liquid import Pipette, Well
from .scene import ARMS, Rack, _yaw_quat

PLATE_OPENING, PLATE_WALL, PLATE_DEPTH, BASE = 0.007, 0.007, 0.011, 0.003
TROUGH_OPENING, TROUGH_WALL, TROUGH_DEPTH = 0.016, 0.004, 0.018
RESERVOIRS = ("stock", "buffer", "sample", "waste")
STANDARDS = {"A1": 1.0, "A2": 0.5, "A3": 0.25, "A4": 0.125, "B1": 0.0625, "B2": 0.0}   # mg/mL
STANDARD_VOLUME, SAMPLE_WELL = 100.0, "B4"
GRIP_ABOVE_TIP = 0.080        # the pads hold the pipette this far above its tip
READY = (-0.03, -0.27, 0.13)  # where the right gripper takes the pipette at the start


@dataclass
class ProteinLayout:
    plate: Rack = field(default_factory=lambda: Rack("plate", (0.06, -0.24), nx=4, ny=2))
    reservoir: Rack = field(default_factory=lambda: Rack("reservoir", (0.02, -0.36), nx=4, ny=1))

    @staticmethod
    def sample(rng: np.random.Generator) -> "ProteinLayout":
        def jitter(name, xy, nx, ny):
            return Rack(name, (xy[0] + rng.uniform(-0.015, 0.015), xy[1] + rng.uniform(-0.015, 0.015)),
                        nx=nx, ny=ny, yaw=rng.uniform(-0.3, 0.3))
        return ProteinLayout(jitter("plate", (0.06, -0.24), 4, 2), jitter("reservoir", (0.02, -0.36), 4, 1))

    @staticmethod
    def cell(rack: Rack, ix: int, iy: int, pitch: float) -> np.ndarray:
        local = np.array([(ix - (rack.nx - 1) / 2) * pitch, (iy - (rack.ny - 1) / 2) * pitch])
        c, s = np.cos(rack.yaw), np.sin(rack.yaw)
        return np.array(rack.xy) + np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])

    def wells(self) -> list[Well]:
        out = []
        for ix, name in enumerate(RESERVOIRS):
            out.append(Well(name, self.cell(self.reservoir, ix, 0, TROUGH_OPENING + TROUGH_WALL), BASE,
                            TROUGH_OPENING / 2, TROUGH_DEPTH, yaw=self.reservoir.yaw))
        for iy, row in enumerate("AB"):
            for ix in range(4):
                out.append(Well(f"{row}{ix + 1}", self.cell(self.plate, ix, iy, PLATE_OPENING + PLATE_WALL), BASE,
                                PLATE_OPENING / 2, PLATE_DEPTH, yaw=self.plate.yaw))
        return out

    def add_to(self, spec: mujoco.MjSpec) -> None:
        _add_block(spec, self.plate, PLATE_OPENING, PLATE_WALL, PLATE_DEPTH, [0.92, 0.92, 0.9, 1])
        _add_block(spec, self.reservoir, TROUGH_OPENING, TROUGH_WALL, TROUGH_DEPTH, [0.75, 0.8, 0.85, 1])
        add_liquids_and_pipette(spec, self.wells())


def add_liquids_and_pipette(spec: mujoco.MjSpec, wells: list[Well], target=(0.03, -0.30, 0.02)) -> None:
    """Liquid shown as a box whose height follows the volume, the pipette, and a close camera."""
    for well in wells:
        spec.worldbody.add_geom(name=f"liquid_{well.name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[well.half * 0.98, well.half * 0.98, 1e-4], pos=[*well.center, well.floor],
                                quat=_yaw_quat(well.yaw), contype=0, conaffinity=0, rgba=[0.6, 0.7, 0.9, 0.0])
    _add_pipette(spec)
    eye, target = np.array([0.36, -0.06, 0.30]), np.array(target)
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    spec.worldbody.add_camera(name="bench", pos=eye, xyaxes=[*right, *np.cross(right, forward)], fovy=45)


def _add_block(spec: mujoco.MjSpec, rack: Rack, opening: float, wall: float, depth: float, rgba) -> None:
    """A block with a grid of square wells: a base plus walls, like the racks but at its own pitch."""
    pitch = opening + wall
    body = spec.worldbody.add_body(name=rack.name, pos=[*rack.xy, 0], quat=_yaw_quat(rack.yaw))
    lx, ly = rack.nx * pitch + wall, rack.ny * pitch + wall
    box = dict(type=mujoco.mjtGeom.mjGEOM_BOX, rgba=rgba)
    body.add_geom(size=[lx / 2, ly / 2, BASE / 2], pos=[0, 0, BASE / 2], **box)
    zc = BASE + depth / 2
    for i in range(rack.nx + 1):
        body.add_geom(size=[wall / 2, ly / 2, depth / 2], pos=[-lx / 2 + wall / 2 + i * pitch, 0, zc], **box)
    for j in range(rack.ny + 1):
        body.add_geom(size=[lx / 2, wall / 2, depth / 2], pos=[0, -ly / 2 + wall / 2 + j * pitch, zc], **box)


def _add_pipette(spec: mujoco.MjSpec) -> None:
    """A pipette: thin shaft (45 mm) under a 16 mm body. The body origin is the tip. Starts lying out of the way."""
    body = spec.worldbody.add_body(name="pipette", pos=[0.33, 0.0, 0.009], quat=[0.7071, 0.7071, 0, 0])
    body.add_freejoint(name="pipette_free")
    cyl = dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER, friction=[1.5, 0.01, 0.0005], condim=4)
    body.add_geom(name="pipette_shaft", size=[0.002, 0.0225, 0], pos=[0, 0, 0.0225], mass=0.003,
                  rgba=[0.95, 0.95, 0.9, 1], **cyl)
    body.add_geom(name="pipette_body", size=[0.008, 0.0375, 0], pos=[0, 0, 0.0825], mass=0.02,
                  rgba=[0.2, 0.45, 0.8, 1], **cyl)


@dataclass
class ProteinTask:
    stock: float          # mg/mL, told to the policy
    sample: float         # mg/mL, hidden: what the policy must measure
    curve: tuple          # (blank, gain, half-saturation) of the absorbance curve, hidden

    @property
    def instruction(self) -> str:
        return (f"Make the protein standards in the plate from the {self.stock:.2f} mg/mL stock and buffer "
                f"({STANDARD_VOLUME:.0f} uL each), put {STANDARD_VOLUME:.0f} uL of the unknown sample in well "
                f"{SAMPLE_WELL}, read the plate, and report the sample's concentration.")


class PipetteEnv(TubeRackEnv):
    """Base for pipetting tasks: wells from the layout, a pipette handed to the right gripper at reset."""

    PIPETTE_CAPACITY = 200.0

    def _fill(self) -> None:
        """Put the starting liquids in the wells."""

    def reset(self, seed: int | None = None) -> dict:
        super().reset(seed)
        self.wells = {w.name: w for w in self.layout.wells()}
        self._fill()
        self.pipette = Pipette(capacity=self.PIPETTE_CAPACITY)
        # Hand the pipette to the right gripper, held straight down at READY.
        self.set_gripper("right", 0.05)
        self.move_to("right", np.array(READY), 0.0)
        m, d = self.model, self.data
        adr, dof = m.joint("pipette_free").qposadr[0], m.joint("pipette_free").dofadr[0]
        d.qpos[adr:adr + 3] = np.array(READY) - [0, 0, GRIP_ABOVE_TIP]
        d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        d.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(m, d)
        self.set_gripper("right", 0.0, wait=0.8)
        self._sync_liquids()
        return self.perceive()

    def tip(self) -> np.ndarray:
        return self.data.body("pipette").xpos.copy()

    def holding_pipette(self, arm: str = "right") -> bool:
        m, d = self.model, self.data
        pipette = m.body("pipette").id
        names = set()
        for c in d.contact[: d.ncon]:
            pair = {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]}
            if pipette in pair:
                names |= {m.body(b).name for b in pair - {pipette}}
        return any(n.startswith(f"{arm}/lf") for n in names) and any(n.startswith(f"{arm}/rf") for n in names)

    def aspirate(self, volume: float) -> dict:
        if not self.holding_pipette():
            return {"volume_ul": 0.0, "note": "the right gripper is not holding the pipette"}
        result = self.pipette.aspirate(list(self.wells.values()), self.tip(), float(volume))
        self.step(0.4)                       # the plunger takes a moment
        self._sync_liquids()
        return result

    def dispense(self, volume: float) -> dict:
        if not self.holding_pipette():
            return {"volume_ul": 0.0, "note": "the right gripper is not holding the pipette"}
        result = self.pipette.dispense(list(self.wells.values()), self.tip(), float(volume))
        self.step(0.4)
        self._sync_liquids()
        return result

    def _liquid_rgba(self, well: Well) -> list[float]:
        return [0.85, 0.3, 0.35, 0.85]

    def _sync_liquids(self) -> None:
        m = self.model
        for well in self.wells.values():
            g = m.geom(f"liquid_{well.name}").id
            height = max(1e-4, (well.surface - well.floor) / 2)
            m.geom_size[g][2] = height
            m.geom_pos[g][2] = well.floor + height
            m.geom_rgba[g] = self._liquid_rgba(well) if well.volume > 0 else [0, 0, 0, 0]

    def perceive(self) -> dict:
        wells = getattr(self, "wells", {})
        labware = {name: {"center": np.array([*w.center, w.floor + w.depth]), "floor_z": w.floor, "depth": w.depth,
                          "opening": 2 * w.half, "area_mm2": w.area_mm2, "volume_ul": round(w.volume, 2)}
                   for name, w in wells.items()}
        pipette = {"tip": self.tip(), "volume_ul": round(self.pipette.volume, 2) if hasattr(self, "pipette") else 0.0,
                   "capacity_ul": self.PIPETTE_CAPACITY}
        return {"labware": labware, "pipette": pipette, "ee": {a: self.ee_pose(a) for a in ARMS}}


class ProteinEnv(PipetteEnv):
    """One arm, a pipette, a plate reader: measure an unknown protein concentration against a standard curve."""

    SIM_BUDGET_S = 300.0

    def _new_episode(self, rng: np.random.Generator) -> None:
        self.layout = ProteinLayout.sample(rng) if self.randomize else ProteinLayout()
        self.task = ProteinTask(round(float(rng.uniform(1.6, 2.4)), 2), float(rng.uniform(0.15, 0.85)),
                                (float(rng.uniform(0.42, 0.48)), float(rng.uniform(1.0, 1.4)),
                                 float(rng.uniform(0.7, 1.1))))
        self._noise = np.random.default_rng(rng.integers(1 << 31))
        self.reported = None

    def _fill(self) -> None:
        for name, volume, conc in (("stock", 1500, self.task.stock), ("buffer", 2500, 0.0),
                                   ("sample", 800, self.task.sample)):
            self.wells[name].add(volume, conc)

    def _liquid_rgba(self, well: Well) -> list[float]:
        shade = well.conc / (well.conc + 0.5)        # Bradford dye: brown without protein, blue with it
        return [0.62 - 0.45 * shade, 0.45 - 0.1 * shade, 0.25 + 0.6 * shade, 0.85]

    def read_plate(self) -> dict:
        blank, gain, half = self.task.curve
        out = {}
        for name, well in self.wells.items():
            if len(name) != 2:
                continue
            if well.volume < 50:
                out[name] = None
            else:
                out[name] = round(blank + gain * well.conc / (well.conc + half) + self._noise.normal(0, 0.005), 4)
        self.step(2.0)                       # the reader takes a couple of seconds
        return out

    def video_status(self) -> list[str]:
        """Line shown under the title in videos: pipette contents and the report so far."""
        report = "-" if self.reported is None else f"{self.reported:.3f} mg/mL (true {self.task.sample:.3f})"
        return [f"pipette {self.pipette.volume:5.1f} uL   reported {report}"]

    def evaluate(self) -> dict:
        return evaluate_protein(self)


def evaluate_protein(env: ProteinEnv) -> dict:
    """Standards right, the sample loaded on its own, and the reported concentration within 10 percent."""
    task, wells = env.task, env.wells
    wrong = []
    for name, target in STANDARDS.items():
        w = wells[name]
        conc_ok = w.conc <= 0.005 if target == 0 else abs(w.conc - target) <= 0.1 * target
        if abs(w.volume - STANDARD_VOLUME) > 10 or not conc_ok:
            wrong.append(f"{name} {w.conc:.3f} mg/mL in {w.volume:.0f} uL (want {target})")
    s = wells[SAMPLE_WELL]
    sample_ok = abs(s.volume - STANDARD_VOLUME) <= 10 and abs(s.conc - task.sample) <= 0.02 * task.sample
    error = None if env.reported is None else (env.reported - task.sample) / task.sample
    checks = {"standards_correct": not wrong, "sample_loaded": sample_ok,
              "estimate_within_10_percent": error is not None and abs(error) <= 0.10}
    success = all(checks.values())
    closeness = 0.0 if error is None else max(0.0, 1.0 - abs(error) / 0.3)
    progress = 1.0 if success else min(0.99, 0.5 * (len(STANDARDS) - len(wrong)) / len(STANDARDS)
                                       + 0.2 * sample_ok + 0.3 * closeness)
    note = (f"reported {'nothing' if env.reported is None else f'{env.reported:.3f} mg/mL'}, true {task.sample:.3f}; "
            f"sample well {s.volume:.0f} uL; {len(STANDARDS) - len(wrong)} of {len(STANDARDS)} standards right"
            + (f" (wrong: {'; '.join(wrong[:3])})" if wrong else "") + f"; {env.pipette.spilled:.0f} uL spilled")
    return {"success": success, "checks": checks, "progress": round(progress, 3), "note": note}


class ProteinApi(TubeRackApi):
    """The right arm, a pipette it already holds, the labware and a plate reader."""

    def functions(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in (
            "get_task", "get_labware", "get_pipette", "get_gripper_pose", "move_to", "set_gripper",
            "aspirate", "dispense", "read_plate", "report_concentration", "wait")}

    def get_task(self) -> dict:
        """Get the goal of this episode.

        Returns:
            dict with keys:
              "instruction": str, the goal in words.
              "stock_mg_per_ml": float, concentration of the protein stock.
              "standards": dict mapping a plate well name to the concentration in mg/mL it
                must hold, for example {"A1": 1.0, ..., "B2": 0.0}.
              "volume_ul": float, the volume every standard and the sample well must hold.
              "sample_well": str, the plate well for the unknown sample.
        """
        t = self.sim.task
        return {"instruction": t.instruction, "stock_mg_per_ml": t.stock, "standards": dict(STANDARDS),
                "volume_ul": STANDARD_VOLUME, "sample_well": SAMPLE_WELL}

    def get_labware(self) -> dict:
        """Get every well you can pipette into or out of.

        A reservoir block holds four square troughs (16 mm wide, 18 mm deep):
        "stock" (protein at the stock concentration), "buffer" (no protein),
        "sample" (the unknown) and "waste" (empty). A plate holds 8 square
        wells (7 mm wide, 11 mm deep, 14 mm apart) named "A1" to "A4" and "B1"
        to "B4". Liquid fills a well from its floor up: its surface is at
        floor_z + volume_ul / area_mm2 / 1000 metres.

        Returns:
            dict mapping each well name to a dict with "center" ((3,) XYZ in metres of
            the middle of the well's opening at its rim), "floor_z", "depth" and
            "opening" (metres), "area_mm2", and "volume_ul" (liquid in it now).
        """
        return {k: {**v, "center": v["center"].copy()} for k, v in self.sim.perceive()["labware"].items()}

    def get_pipette(self) -> dict:
        """Get the pipette's tip position and what it holds.

        The right gripper holds the pipette straight down from the start, its pads
        80 mm above the tip. The pipette is a 45 mm shaft, 4 mm wide, under a
        16 mm wide body. If the gripper opens, the pipette falls.

        Returns:
            dict with "tip" ((3,) XYZ in metres), "volume_ul" (liquid in the pipette)
            and "capacity_ul" (200).
        """
        p = self.sim.perceive()["pipette"]
        return {"tip": p["tip"].copy(), "volume_ul": p["volume_ul"], "capacity_ul": p["capacity_ul"]}

    def move_to(self, arm: str, position: np.ndarray, yaw: float = 0.0, tilt: float = 0.0,
                speed: float = 0.15) -> bool:
        """Move the pinch point of an arm's gripper in a straight line to a position.

        World frame: metres, Z up, table top at z = 0. The right arm's base is at
        (-0.20, -0.28, 0), facing +X; the plate and the reservoir are within its
        reach. Keep the gripper straight down (tilt 0) while it holds the pipette.

        Args:
            arm: "left" or "right".
            position: (3,) XYZ target of the pinch point in metres.
            yaw: rotation of the gripper about world Z in radians.
            tilt: 0 points the gripper straight down.
            speed: metres per second. Use about 0.03 to 0.05 near and inside wells.

        Returns:
            True if the target was reached within 1 mm. False if it is unreachable,
            in which case the arm does not move, or if the arm was blocked on the way,
            for example by the pipette tip pressing on a well's floor or wall.
        """
        return bool(self.sim.move_to(arm, np.asarray(position, dtype=float), yaw, tilt, speed))

    def aspirate(self, volume_ul: float) -> dict:
        """Draw liquid into the pipette from the well its tip is in.

        Liquid is drawn only if the tip is inside a well's opening and at least
        0.5 mm below the liquid surface; otherwise the pipette draws air. It
        holds at most 200 uL. Liquid in a well is always evenly mixed.

        Args:
            volume_ul: microlitres to draw.

        Returns:
            dict with "volume_ul" (liquid actually drawn), "from" (well name or None)
            and, when something went wrong, "note".
        """
        return self.sim.aspirate(float(volume_ul))

    def dispense(self, volume_ul: float) -> dict:
        """Push liquid out of the pipette into the well its tip is in.

        With the tip outside every well the liquid is spilled and lost.

        Args:
            volume_ul: microlitres to dispense (at most what the pipette holds).

        Returns:
            dict with "volume_ul" (liquid that went into the well), "into" (well name or
            None) and, when something went wrong, "note".
        """
        return self.sim.dispense(float(volume_ul))

    def read_plate(self) -> dict:
        """Read the absorbance of every plate well with at least 50 uL in it.

        The dye's absorbance rises with protein concentration and levels off at
        high concentration; its exact curve differs from day to day, so compare
        against standards read in the same run. Readings carry a little noise.

        Returns:
            dict mapping each plate well name to its absorbance, or None if it holds
            under 50 uL.
        """
        return self.sim.read_plate()

    def report_concentration(self, mg_per_ml: float) -> None:
        """Report the unknown sample's protein concentration. The last report counts.

        Args:
            mg_per_ml: your measured concentration of the sample, in mg/mL.
        """
        self.sim.reported = float(mg_per_ml)

    def wait(self, seconds: float) -> None:
        """Hold still and let time pass.

        Args:
            seconds: how long to wait, at most 5.
        """
        self.sim.step(float(np.clip(seconds, 0.0, 5.0)))


PROTEIN_PROMPT = """\
You control two I2RT YAM robot arms on a lab bench through the Python API below; this task uses the right arm only.
Task: run a protein assay. The right gripper already holds a pipette. Make a dilution series of a protein standard in a plate from a stock and buffer, load the unknown sample, read the plate, and report the sample's concentration. Call get_task() for the stock concentration and the standards to make; the stock concentration, the positions of the plate and the reservoir, and the dye's response all differ in every episode.
An episode succeeds when every standard well holds the right concentration (within 10 percent) and volume (within 10 uL), the sample well holds the sample alone, and the reported concentration is within 10 percent of the truth."""


PROTEIN_REFERENCE_POLICY = """
import numpy as np

task = get_task()
labware = get_labware()
offset = get_gripper_pose("right")["pos"] - get_pipette()["tip"]     # pinch point relative to the tip
TRAVEL = 0.05                                                        # tip height when moving between wells

def tip_to(point, speed=0.1):
    return move_to("right", np.asarray(point, dtype=float) + offset, 0.0, speed=speed)

def go_into(well, height):
    w = labware[well]
    tip_to([w["center"][0], w["center"][1], TRAVEL])
    tip_to([w["center"][0], w["center"][1], w["floor_z"] + height], speed=0.04)

def leave(well):
    w = labware[well]
    tip_to([w["center"][0], w["center"][1], TRAVEL], speed=0.06)

def transfer(source, target, volume):
    go_into(source, 0.001)                     # the tip near the floor stays under the surface
    got = aspirate(volume)["volume_ul"]
    leave(source)
    go_into(target, 0.004)
    dispense(got)
    leave(target)
    return got

standards = task["standards"]                                     # A1 1.0, A2 0.5, ... B1 0.0625, B2 0.0
series = [name for name, c in sorted(standards.items(), key=lambda kv: -kv[1]) if c > 0]
top = series[0]
stock_ul = 2 * task["volume_ul"] * standards[top] / task["stock_mg_per_ml"]
transfer("stock", top, stock_ul)                                   # 200 uL of the top standard
transfer("buffer", top, 2 * task["volume_ul"] - stock_ul)
for name in series[1:] + [n for n, c in standards.items() if c == 0]:
    transfer("buffer", name, task["volume_ul"])
for here, there in zip(series, series[1:]):                         # halve down the series
    transfer(here, there, task["volume_ul"])
transfer(series[-1], "waste", task["volume_ul"])
transfer("sample", task["sample_well"], task["volume_ul"])

reading = read_plate()
conc = np.array([standards[n] for n in standards])
absorb = np.array([reading[n] for n in standards])
# The response levels off, so fit A = blank + gain * c / (c + half) rather than joining the dots:
# for each candidate half-saturation the other two follow by least squares; keep the best fit.
best = None
for half in np.linspace(0.1, 3.0, 300):
    basis = np.column_stack([np.ones_like(conc), conc / (conc + half)])
    coef, *_ = np.linalg.lstsq(basis, absorb, rcond=None)
    error = float(np.sum((basis @ coef - absorb) ** 2))
    if best is None or error < best[0]:
        best = (error, half, coef)
_, half, (blank, gain) = best
signal = reading[task["sample_well"]] - blank
estimate = float(half * signal / (gain - signal))
report_concentration(estimate)
print("fit blank", round(blank, 3), "gain", round(gain, 3), "half", round(half, 3), "sample", reading[task["sample_well"]],
      "estimate", round(estimate, 3))
"""
