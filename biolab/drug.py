"""Drug dose-response: seed cancer cells in a plate, treat them with a dilution series, find the IC50.

Combines the protein assay's pipetting (biolab.protein) with a cell model like the culture task's
(biolab.culture). Cells sit in an assay plate; a drug slows their growth and kills them more the higher its
concentration, following a Hill curve whose midpoint and steepness are hidden. After treatment a plate
reader measures a signal proportional to the living cells. The policy reports the IC50: the drug
concentration that halves the signal of untreated cells after 48 hours, with a cell-free blank subtracted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .liquid import Well
from .protein import (BASE, PLATE_DEPTH, PLATE_OPENING, PLATE_WALL, TROUGH_DEPTH, TROUGH_OPENING, TROUGH_WALL,
                      PipetteEnv, ProteinApi, ProteinLayout, _add_block, add_liquids_and_pipette)
from .scene import Rack

RESERVOIRS = ("cells", "medium", "drug", "waste")
CELLS_PER_UL = 50.0           # the suspension: 100 uL seeds 5000 cells
STOCK_UM = 100.0              # drug stock, micromolar
SEED, SETTLE, EXPOSURE = 5000.0, 24.0, 48.0
WELL_CAP = 3.0e4              # cells a well can hold
STEP = 0.1                    # hours per biology step
DEATH = 0.035                 # per hour, at full drug effect
BACKGROUND, PER_CELL = 800.0, 0.06


@dataclass
class DrugLayout:
    assay: Rack = field(default_factory=lambda: Rack("assay_plate", (0.03, -0.215), nx=4, ny=3))
    dilution: Rack = field(default_factory=lambda: Rack("dilution_plate", (0.085, -0.325), nx=4, ny=2))
    reservoir: Rack = field(default_factory=lambda: Rack("reservoir", (0.0, -0.385), nx=4, ny=1))

    @staticmethod
    def sample(rng: np.random.Generator) -> "DrugLayout":
        def jitter(name, xy, nx, ny):
            return Rack(name, (xy[0] + rng.uniform(-0.01, 0.01), xy[1] + rng.uniform(-0.01, 0.01)),
                        nx=nx, ny=ny, yaw=rng.uniform(-0.2, 0.2))
        return DrugLayout(jitter("assay_plate", (0.03, -0.215), 4, 3), jitter("dilution_plate", (0.085, -0.325), 4, 2),
                          jitter("reservoir", (0.0, -0.385), 4, 1))

    def wells(self) -> list[Well]:
        cell, out = ProteinLayout.cell, []
        for ix, name in enumerate(RESERVOIRS):
            out.append(Well(name, cell(self.reservoir, ix, 0, TROUGH_OPENING + TROUGH_WALL), BASE, TROUGH_OPENING / 2,
                            TROUGH_DEPTH, yaw=self.reservoir.yaw))
        for iy, row in enumerate("ABC"):
            for ix in range(4):
                out.append(Well(f"{row}{ix + 1}", cell(self.assay, ix, iy, PLATE_OPENING + PLATE_WALL), BASE,
                                PLATE_OPENING / 2, PLATE_DEPTH, yaw=self.assay.yaw))
        for i in range(8):
            out.append(Well(f"D{i + 1}", cell(self.dilution, i % 4, i // 4, PLATE_OPENING + PLATE_WALL), BASE,
                            PLATE_OPENING / 2, PLATE_DEPTH, yaw=self.dilution.yaw))
        return out

    def add_to(self, spec) -> None:
        _add_block(spec, self.assay, PLATE_OPENING, PLATE_WALL, PLATE_DEPTH, [0.92, 0.92, 0.9, 1])
        _add_block(spec, self.dilution, PLATE_OPENING, PLATE_WALL, PLATE_DEPTH, [0.85, 0.9, 0.95, 1])
        _add_block(spec, self.reservoir, TROUGH_OPENING, TROUGH_WALL, TROUGH_DEPTH, [0.75, 0.8, 0.85, 1])
        add_liquids_and_pipette(spec, self.wells(), target=(0.04, -0.30, 0.02))


@dataclass
class DrugTask:
    ic50: float          # micromolar, hidden: the midpoint of the drug's effect
    hill: float          # hidden steepness
    doubling: float      # hours, hidden

    @property
    def instruction(self) -> str:
        return ("Measure the drug's IC50 on these cancer cells: seed about 5000 cells in 100 uL per well, let them "
                "settle 24 h, add the drug at concentrations of your choosing (final volume 200 uL per well), "
                "incubate 48 h, read the plate, and report the concentration that halves the signal of untreated "
                "cells after subtracting a cell-free blank.")


def grow(cells: float, conc: float, task: DrugTask, hours: float) -> tuple[float, float]:
    """Advance one well's cells; returns (living, newly dead)."""
    rate = np.log(2) / task.doubling
    effect = conc ** task.hill / (conc ** task.hill + task.ic50 ** task.hill) if conc > 0 else 0.0
    dead = 0.0
    for _ in range(int(round(hours / STEP))):
        born = rate * (1 - effect) * cells * (1 - cells / WELL_CAP) * STEP
        died = DEATH * effect * cells * STEP
        cells, dead = cells + born - died, dead + died
    return cells, dead


def true_ic50(task: DrugTask) -> float:
    """The concentration whose 48 h signal (blank subtracted) is half the untreated one, by bisection."""
    settled, _ = grow(SEED, 0.0, task, SETTLE)
    control, _ = grow(settled, 0.0, task, EXPOSURE)
    lo, hi = np.log(1e-4), np.log(1e3)
    for _ in range(60):
        mid = (lo + hi) / 2
        treated, _ = grow(settled, float(np.exp(mid)), task, EXPOSURE)
        lo, hi = (mid, hi) if treated > 0.5 * control else (lo, mid)
    return float(np.exp((lo + hi) / 2))


class DrugEnv(PipetteEnv):
    """Assay plate, dilution plate, reservoirs of cells, medium, drug and waste, an incubator and a plate reader."""

    SIM_BUDGET_S = 600.0
    VIDEO_EVERY = 25

    def _new_episode(self, rng: np.random.Generator) -> None:
        self.layout = DrugLayout.sample(rng) if self.randomize else DrugLayout()
        self.task = DrugTask(float(np.exp(rng.uniform(np.log(0.05), np.log(5.0)))), float(rng.uniform(0.8, 2.0)),
                             float(rng.uniform(20, 28)))
        self._noise = np.random.default_rng(rng.integers(1 << 31))
        self.clock, self.reported, self.reads = 0.0, None, 0

    def _fill(self) -> None:
        self.wells["cells"].add(3000, 0.0, {"cells": 3000 * CELLS_PER_UL})
        self.wells["medium"].add(4500, 0.0)
        self.wells["drug"].add(1500, STOCK_UM)

    def _liquid_rgba(self, well: Well) -> list[float]:
        if well.name == "waste":
            return [0.6, 0.55, 0.4, 0.8]
        if well.extra.get("cells", 0) > 0 and well.volume > 0:
            return [0.9, 0.4, 0.55, 0.85]        # cells in medium
        shade = min(1.0, well.conc / 30.0)
        return [0.95 - 0.3 * shade, 0.75 - 0.4 * shade, 0.2 + 0.1 * shade, 0.85]   # drug turns it amber

    def incubate(self, hours: float) -> None:
        hours = float(np.clip(hours, 0.0, 72.0))
        for well in self.wells.values():
            cells = well.extra.get("cells", 0.0)
            if cells > 0 and well.volume > 0:
                living, dead = grow(cells, well.conc, self.task, hours)
                well.extra["cells"], well.extra["dead"] = living, well.extra.get("dead", 0.0) + dead
        self.clock += hours
        self.step(0.5)
        self._sync_liquids()

    def read_plate(self) -> dict:
        out = {}
        for name, well in self.wells.items():
            if len(name) != 2 or name[0] == "D":
                continue
            if well.volume < 50:
                out[name] = None
            else:
                signal = BACKGROUND + PER_CELL * well.extra.get("cells", 0.0)
                out[name] = round(signal * (1 + self._noise.normal(0, 0.02)) + self._noise.normal(0, 15), 1)
        self.reads += 1
        self.step(2.0)
        return out

    def evaluate(self) -> dict:
        return evaluate_drug(self)


def evaluate_drug(env: DrugEnv) -> dict:
    truth = true_ic50(env.task)
    ratio = None if not env.reported or env.reported <= 0 else env.reported / truth
    seeded = sum(1 for n, w in env.wells.items() if len(n) == 2 and n[0] != "D" and w.extra.get("cells", 0) > 1000)
    dosed = sum(1 for n, w in env.wells.items() if len(n) == 2 and n[0] != "D" and w.conc > 0)
    checks = {"ic50_within_1.5_fold": ratio is not None and 1 / 1.5 <= ratio <= 1.5}
    success = all(checks.values())
    closeness = 0.0 if ratio is None else max(0.0, 1.0 - abs(np.log10(ratio)))
    progress = 1.0 if success else min(0.99, 0.1 * (seeded >= 3) + 0.1 * (dosed >= 3) + 0.1 * (env.reads > 0)
                                       + 0.7 * closeness)
    note = (f"reported {'nothing' if ratio is None else f'{env.reported:.3g} uM'}, true IC50 {truth:.3g} uM"
            + ("" if ratio is None else f" ({ratio:.2f} x)") + f"; {seeded} wells seeded, {dosed} dosed; clock "
            f"{env.clock:.0f} h; plate read {env.reads} times; {env.pipette.spilled:.0f} uL spilled")
    return {"success": success, "checks": checks, "progress": round(progress, 3), "note": note}


class DrugApi(ProteinApi):
    """The right arm with a 200 uL pipette, two plates, four reservoirs, an incubator and a plate reader."""

    def functions(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in (
            "get_task", "get_labware", "get_pipette", "get_gripper_pose", "move_to", "set_gripper",
            "aspirate", "dispense", "incubate", "get_clock", "read_plate", "report_ic50")}

    def get_task(self) -> dict:
        """Get the goal of this episode.

        Returns:
            dict with keys "instruction" (str), "cells_per_ul" (cells in the
            suspension per microlitre), "drug_stock_um" (drug stock concentration,
            micromolar), "seed_cells" (cells to seed per well), "settle_h" and
            "exposure_h" (hours before and after adding the drug).
        """
        return {"instruction": self.sim.task.instruction, "cells_per_ul": CELLS_PER_UL, "drug_stock_um": STOCK_UM,
                "seed_cells": SEED, "settle_h": SETTLE, "exposure_h": EXPOSURE}

    def get_labware(self) -> dict:
        """Get every well you can pipette into or out of.

        Four reservoir troughs (16 mm wide, 18 mm deep): "cells" (3 mL of cell
        suspension), "medium" (4.5 mL, no drug), "drug" (1.5 mL of drug stock)
        and "waste" (empty). An assay plate of 12 wells "A1" to "C4" and a
        dilution plate of 8 wells "D1" to "D8", all 7 mm square, 11 mm deep and
        14 mm apart, holding up to about 530 uL. Only the assay plate is read.
        Liquid is always evenly mixed, including the drug and any cells; liquid
        fills a well from its floor up: its surface is at
        floor_z + volume_ul / area_mm2 / 1000 metres.

        Returns:
            dict mapping each well name to a dict with "center" ((3,) XYZ in metres of
            the middle of the opening at its rim), "floor_z", "depth", "opening"
            (metres), "area_mm2" and "volume_ul".
        """
        return super().get_labware()

    def incubate(self, hours: float) -> None:
        """Put the plates in the incubator; the clock moves on. Robot work takes no culture time.

        Cells grow in any well with medium; the drug slows their growth and kills
        them, more strongly at higher concentration. How fast the cells grow and
        how the drug acts differ from episode to episode.

        Args:
            hours: how long, at most 72 per call.
        """
        self.sim.incubate(float(hours))

    def get_clock(self) -> float:
        """Get the clock in hours since the start of the episode."""
        return float(self.sim.clock)

    def read_plate(self) -> dict:
        """Read every assay-plate well with at least 50 uL in it (takes two seconds).

        The signal is proportional to the living cells in a well, on top of a
        constant background, with about 2 percent noise.

        Returns:
            dict mapping each assay well name to its signal, or None if it holds
            under 50 uL.
        """
        return self.sim.read_plate()

    def report_ic50(self, micromolar: float) -> None:
        """Report the drug's IC50 in micromolar. The last report counts.

        Args:
            micromolar: the concentration that halves the blank-subtracted signal of
              untreated cells after 48 h of exposure.
        """
        self.sim.reported = float(micromolar)


DRUG_PROMPT = """\
You control two I2RT YAM robot arms on a lab bench through the Python API below; this task uses the right arm only.
Task: measure how potent a drug is on cancer cells. The right gripper already holds a 200 uL pipette. Seed cells in an assay plate, let them settle, treat them with a range of drug concentrations you prepare by dilution, incubate, read the plate, and report the IC50: the concentration that halves the signal of untreated cells after 48 hours, with a cell-free blank subtracted. Call get_task() for the protocol. The drug's potency (anywhere from about 0.05 to 5 micromolar), how steeply it acts, how fast the cells grow, and the positions of the plates and reservoirs differ in every episode.
An episode succeeds when the reported IC50 is within a factor of 1.5 of the true one."""


DRUG_REFERENCE_POLICY = """
import numpy as np

task = get_task()
labware = get_labware()
TRAVEL = 0.05

def tip_to(point, speed=0.12):
    # Measure where the tip sits under the pads before every move: it can slide in the grip.
    offset = get_gripper_pose("right")["pos"] - get_pipette()["tip"]
    return move_to("right", np.asarray(point, dtype=float) + offset, 0.0, speed=speed)

def go_into(well, height):
    w = labware[well]
    tip_to([w["center"][0], w["center"][1], TRAVEL])
    tip_to([w["center"][0], w["center"][1], w["floor_z"] + height], speed=0.05)

def leave(well):
    w = labware[well]
    tip_to([w["center"][0], w["center"][1], TRAVEL], speed=0.08)

def transfer(source, target, volume):
    moved = 0.0
    while volume - moved > 0.5:
        go_into(source, 0.0005)
        got = aspirate(min(200.0, volume - moved))["volume_ul"]
        if got <= 0:                                    # drew air: the tip stopped short; press to the floor
            go_into(source, -0.0005)
            got = aspirate(min(200.0, volume - moved))["volume_ul"]
        leave(source)
        if got <= 0:
            break
        go_into(target, 0.004)
        dispense(got)
        leave(target)
        moved += got
    return moved

assay = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4"]
treated, controls, blank = assay[:8], ["C1", "C2", "C3"], "C4"
seed_ul = task["seed_cells"] / task["cells_per_ul"]
for well in treated + controls:
    transfer("cells", well, seed_ul)
transfer("medium", blank, 100)
incubate(task["settle_h"])

# Twice the final concentrations, 3-fold apart from 30 uM down, made in the dilution plate (300 uL each).
dilutions = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
top = 2 * 30.0
stock_ul = 300 * top / task["drug_stock_um"]
transfer("drug", "D1", stock_ul)
transfer("medium", "D1", 300 - stock_ul)
for well in dilutions[1:]:
    transfer("medium", well, 200)
for here, there in zip(dilutions, dilutions[1:]):
    transfer(here, there, 100)
finals = [30.0 / 3 ** i for i in range(8)]
for dil, well in zip(dilutions, treated):
    transfer(dil, well, 100)                      # halves it: the final concentrations are `finals`
for well in controls + [blank]:
    transfer("medium", well, 100)
incubate(task["exposure_h"])

reading = read_plate()
background = reading[blank]
control = np.mean([reading[w] for w in controls]) - background
fraction = np.array([(reading[w] - background) / control for w in treated])   # high dose first
logc = np.log10(finals)
# Fit a falling Hill curve on log-concentration and solve it for one half.
best = None
for mid in np.linspace(-2.5, 1.7, 211):
    for slope in np.linspace(0.3, 4.0, 38):
        model = 1 / (1 + 10 ** (slope * (logc - mid)))
        error = float(np.sum((model - fraction) ** 2))
        if best is None or error < best[0]:
            best = (error, mid, slope)
ic50 = 10 ** best[1]
report_ic50(ic50)
print("fraction of control", np.round(fraction, 3).tolist(), "IC50", round(float(ic50), 4))
"""
