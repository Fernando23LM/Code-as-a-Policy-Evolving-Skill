"""Cell culture: keep a cancer cell line growing for a simulated week and finish with enough healthy cells.

The robot work is real simulation (the right arm pipettes between three culture dishes and three
reservoirs: fresh medium, trypsin and waste). The cells are a small model stepped forward only when the
policy incubates: they grow logistically towards the dish's capacity while the medium lasts, starve when
it is used up, suffer when overgrown, come off the dish in trypsin and settle again once the trypsin is
diluted. Doubling time and nutrient use differ from episode to episode and are not told, so the policy
has to look at the dishes and decide when to feed and when to split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .liquid import Well
from .protein import BASE, ProteinApi, PipetteEnv, _add_block, add_liquids_and_pipette
from .scene import Rack

DISH_OPENING, DISH_WALL, DISH_DEPTH = 0.024, 0.006, 0.012     # 2 mL fills a dish 3.5 mm deep
RES_OPENING, RES_WALL, RES_DEPTH = 0.030, 0.005, 0.040     # 36 mL each; the pipette body clears the rim
DISHES, RESERVOIRS = ("A", "B", "C"), ("medium", "trypsin", "waste")
CAPACITY = 1.0e6              # cells in a confluent dish
WEEK, LATEST = 168.0, 180.0   # hours: the culture is judged between these clock times
STEP = 0.05                   # hours per biology step
FEED_MIN = 0.15               # nutrient fraction below which cells starve


@dataclass
class CultureLayout:
    dishes: Rack = field(default_factory=lambda: Rack("dishes", (0.06, -0.21), nx=3, ny=1))
    reservoirs: Rack = field(default_factory=lambda: Rack("reservoirs", (0.04, -0.36), nx=3, ny=1))

    @staticmethod
    def sample(rng: np.random.Generator) -> "CultureLayout":
        def jitter(name, xy):
            return Rack(name, (xy[0] + rng.uniform(-0.012, 0.012), xy[1] + rng.uniform(-0.012, 0.012)),
                        nx=3, ny=1, yaw=rng.uniform(-0.25, 0.25))
        return CultureLayout(jitter("dishes", (0.06, -0.21)), jitter("reservoirs", (0.04, -0.36)))

    @staticmethod
    def _cell(rack: Rack, ix: int, pitch: float) -> np.ndarray:
        x = (ix - (rack.nx - 1) / 2) * pitch
        return np.array(rack.xy) + x * np.array([np.cos(rack.yaw), np.sin(rack.yaw)])

    def wells(self) -> list[Well]:
        out = [Well(n, self._cell(self.reservoirs, i, RES_OPENING + RES_WALL), BASE, RES_OPENING / 2, RES_DEPTH,
                    yaw=self.reservoirs.yaw) for i, n in enumerate(RESERVOIRS)]
        out += [Well(n, self._cell(self.dishes, i, DISH_OPENING + DISH_WALL), BASE, DISH_OPENING / 2, DISH_DEPTH,
                     yaw=self.dishes.yaw) for i, n in enumerate(DISHES)]
        return out

    def add_to(self, spec) -> None:
        _add_block(spec, self.dishes, DISH_OPENING, DISH_WALL, DISH_DEPTH, [0.85, 0.88, 0.92, 1])
        _add_block(spec, self.reservoirs, RES_OPENING, RES_WALL, RES_DEPTH, [0.75, 0.8, 0.85, 1])
        add_liquids_and_pipette(spec, self.wells(), target=(0.04, -0.29, 0.02))


@dataclass
class CultureTask:
    target: float           # viable attached cells wanted at the end
    doubling: float         # hours, hidden
    uptake: float           # nutrient used per cell per hour (microlitres of fresh medium), hidden

    @property
    def instruction(self) -> str:
        return (f"Grow the cells in dish A for one week: when the clock reads between {WEEK:.0f} and {LATEST:.0f} "
                f"hours the dishes together must hold at least {self.target / 1e6:.1f} million healthy attached "
                "cells, at least 90 percent of the cells alive, none starved or overgrown for long.")


class CultureEnv(PipetteEnv):
    """Three dishes, medium, trypsin, waste, a microscope and an incubator clock."""

    PIPETTE_CAPACITY = 1000.0
    SIM_BUDGET_S = 900.0     # robot time only: incubation does not count
    WALL_BUDGET_S = 900

    def _new_episode(self, rng: np.random.Generator) -> None:
        self.layout = CultureLayout.sample(rng) if self.randomize else CultureLayout()
        self.task = CultureTask(1.8e6, float(rng.uniform(20, 28)), float(rng.uniform(5e-5, 9e-5)))
        self._noise = np.random.default_rng(rng.integers(1 << 31))
        self.clock = 0.0
        self.attached = {d: 0.0 for d in DISHES}
        self.starved = {d: 0.0 for d in DISHES}      # hours spent starving
        self.crowded = {d: 0.0 for d in DISHES}      # hours spent above 95 percent confluence

    def _fill(self) -> None:
        self.wells["medium"].add(34000, 0.0, {"nutrient": 34000.0})     # 1 unit per microlitre when fresh
        self.wells["trypsin"].add(3000, 1.0)            # concentration = trypsin fraction
        self.wells["A"].add(2000, 0.0, {"nutrient": 2000.0})
        self.attached["A"] = 1.5e5

    def _liquid_rgba(self, well: Well) -> list[float]:
        fresh = well.extra.get("nutrient", 0.0) / max(well.volume, 1e-9)
        if well.name in ("trypsin", "waste"):
            return [0.85, 0.85, 0.8, 0.6] if well.name == "trypsin" else [0.6, 0.55, 0.4, 0.8]
        return [0.9, 0.25 + 0.55 * (1 - fresh), 0.3, 0.85]   # phenol red: red when fresh, yellow when spent

    # ---------------------------------------------------------------- biology
    def incubate(self, hours: float) -> None:
        hours = float(np.clip(hours, 0.0, 24.0))
        rate = np.log(2) / self.task.doubling
        for _ in range(int(round(hours / STEP))):
            for d in DISHES:
                self._grow(self.wells[d], d, rate)
            self.clock += STEP
        self.step(0.5)                               # the door opens and closes
        self._sync_liquids()

    def _grow(self, well: Well, name: str, rate: float) -> None:
        n, extra = self.attached[name], well.extra
        floating = extra.get("cells", 0.0)
        if n + floating <= 0:
            return
        dt, vol = STEP, well.volume
        dead = 0.0
        if vol < 200:                                # dried out
            dead += (n + floating) * min(1.0, 0.5 * dt)
            n -= n * min(1.0, 0.5 * dt); floating -= floating * min(1.0, 0.5 * dt)
        nutrient = extra.get("nutrient", 0.0)
        fed = vol > 0 and nutrient / vol > FEED_MIN
        tryp = well.conc if vol > 0 else 0.0
        if tryp > 0.3:                               # trypsin lifts cells off and slowly harms them
            lift = n * min(1.0, 30 * dt)
            n -= lift; floating += lift
            dead += floating * 0.10 * dt; floating -= floating * 0.10 * dt
        else:
            settle = floating * min(1.0, 0.8 * dt) * max(0.0, 1 - n / CAPACITY)
            n += settle; floating -= settle
            dead += floating * 0.03 * dt; floating -= floating * 0.03 * dt
            if fed and n < CAPACITY:
                n += rate * n * (1 - n / CAPACITY) * dt
        if not fed:
            self.starved[name] += dt
            dead += (n + floating) * 0.03 * dt; n -= n * 0.03 * dt; floating -= floating * 0.03 * dt
        if n > 0.95 * CAPACITY:
            self.crowded[name] += dt
            dead += n * 0.01 * dt; n -= n * 0.01 * dt
        extra["nutrient"] = max(0.0, nutrient - self.task.uptake * (n + floating) * dt)
        extra["cells"], extra["dead"] = floating, extra.get("dead", 0.0) + dead
        self.attached[name] = n

    def inspect(self, dish: str) -> dict:
        well = self.wells[dish]
        n = self.attached[dish]
        fresh = well.extra.get("nutrient", 0.0) / max(well.volume, 1e-9) if well.volume > 0 else 0.0
        floating = well.extra.get("cells", 0.0) + well.extra.get("dead", 0.0)
        self.step(1.0)                               # a look under the microscope
        return {"confluence_percent": round(float(np.clip(100 * n / CAPACITY + self._noise.normal(0, 2.5), 0, 100)), 1),
                "attached_cells_estimate": int(max(0.0, n * (1 + self._noise.normal(0, 0.08)))),
                "medium_ul": round(well.volume, 1),
                "medium_color": "red" if fresh > 0.5 else "orange" if fresh > 0.25 else "yellow",
                "floating_cells": "many" if floating > 0.2 * max(n, 1.0) else "few" if floating > 1e3 else "none"}

    def viability(self) -> float:
        alive = sum(self.attached[d] + self.wells[d].extra.get("cells", 0.0) for d in DISHES)
        dead = sum(self.wells[d].extra.get("dead", 0.0) for d in DISHES)
        return alive / (alive + dead) if alive + dead > 0 else 0.0

    def perceive(self) -> dict:
        out = super().perceive()
        out["clock_h"] = round(self.clock, 2)
        return out

    def video_status(self) -> list[str]:
        """Lines shown under the title in videos: the clock and the attached cells per dish."""
        total = sum(self.attached.values())
        dishes = "   ".join(f"{d} {self.attached[d] / 1e6:.2f}M" for d in DISHES)
        return [f"day {int(self.clock // 24) + 1}   clock {self.clock:5.1f} h   cells {total / 1e6:.2f}M "
                f"of {self.task.target / 1e6:.1f}M", dishes]

    def evaluate(self) -> dict:
        return evaluate_culture(self)


def evaluate_culture(env: CultureEnv) -> dict:
    task, clock = env.task, env.clock
    cells = sum(env.attached.values())
    covered = all(env.wells[d].volume >= 1000 for d in DISHES if env.attached[d] > 1e3)
    checks = {
        "week_finished": WEEK <= clock <= LATEST,
        "enough_cells": cells >= task.target,
        "viability_90_percent": env.viability() >= 0.9,
        "never_starved": max(env.starved.values()) <= 6.0,
        "never_overgrown": max(env.crowded.values()) <= 12.0,
        "cells_covered_in_medium": covered,
    }
    success = all(checks.values())
    time_share = min(1.0, clock / WEEK)
    progress = 1.0 if success else min(0.99, time_share * (0.5 * min(1.0, cells / task.target) + 0.2 * (env.viability() >= 0.9)
                                                          + 0.15 * checks["never_starved"] + 0.15 * checks["never_overgrown"]))
    note = (f"clock {clock:.1f} h, {cells / 1e6:.2f} million attached cells (target {task.target / 1e6:.1f}), "
            f"viability {100 * env.viability():.0f}%, per dish " +
            ", ".join(f"{d}: {env.attached[d] / 1e6:.2f}M in {env.wells[d].volume:.0f} uL, starved {env.starved[d]:.0f} h, "
                      f"overgrown {env.crowded[d]:.0f} h" for d in DISHES) + f"; {env.pipette.spilled:.0f} uL spilled")
    return {"success": success, "checks": checks, "progress": round(progress, 3), "note": note}


class CultureApi(ProteinApi):
    """The right arm with a 1 mL pipette, three dishes, three reservoirs, a microscope and an incubator."""

    def functions(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in (
            "get_task", "get_labware", "get_pipette", "get_gripper_pose", "move_to", "set_gripper",
            "aspirate", "dispense", "inspect", "incubate", "get_clock")}

    def get_task(self) -> dict:
        """Get the goal of this episode.

        Returns:
            dict with keys:
              "instruction": str, the goal in words.
              "target_cells": float, healthy attached cells wanted at the end.
              "finish_between_h": (start, end) clock hours in which the episode should end.
        """
        t = self.sim.task
        return {"instruction": t.instruction, "target_cells": t.target, "finish_between_h": (WEEK, LATEST)}

    def get_labware(self) -> dict:
        """Get every well you can pipette into or out of.

        Three culture dishes "A", "B", "C" (square, 24 mm wide, 12 mm deep; a
        dish is full of cells at about 1 million). Dish A starts with about
        150 thousand cells in 2 mL of medium; B and C are empty. Three square
        reservoirs, 30 mm wide and 40 mm deep: "medium" (34 mL of fresh medium
        for the whole week), "trypsin" (3 mL) and "waste" (empty, holds 36 mL). Liquid fills a well from its
        floor up: its surface is at floor_z + volume_ul / area_mm2 / 1000 metres.

        Returns:
            dict mapping each well name to a dict with "center" ((3,) XYZ in metres of
            the middle of the opening at its rim), "floor_z", "depth" and "opening"
            (metres), "area_mm2", and "volume_ul" (liquid in it now).
        """
        return super().get_labware()

    def get_pipette(self) -> dict:
        """Get the pipette's tip position and what it holds.

        The right gripper holds the pipette straight down from the start, its pads
        80 mm above the tip. It is a 1 mL pipette: a 45 mm shaft, 4 mm wide,
        under a 16 mm wide body. If the gripper opens, the pipette falls.

        Returns:
            dict with "tip" ((3,) XYZ in metres), "volume_ul" (liquid in the pipette)
            and "capacity_ul" (1000).
        """
        return super().get_pipette()

    def aspirate(self, volume_ul: float) -> dict:
        """Draw liquid into the pipette from the well its tip is in.

        Liquid is drawn only if the tip is inside a well's opening and at least
        0.5 mm below the surface, or within 0.8 mm of the floor (where it can draw
        the last drops); otherwise it draws air. Liquid is always evenly mixed,
        including any cells floating in it; cells stuck to a dish stay behind.

        Args:
            volume_ul: microlitres to draw (the pipette holds at most 1000).

        Returns:
            dict with "volume_ul" (liquid actually drawn), "from" (well name or None)
            and, when something went wrong, "note".
        """
        return self.sim.aspirate(float(volume_ul))

    def inspect(self, dish: str) -> dict:
        """Look at a dish under the microscope (takes a second of robot time).

        Cells grow stuck to the dish floor. Trypsin lifts them off within a few
        minutes in the incubator (it must make up more than about a third of
        the liquid), but harms them the longer they sit in it; diluting it with
        at least twice as much medium stops it, and the cells then settle and
        stick again over an hour or two. Cells use up their medium, turning it
        from red through orange to yellow, and starve and die when it is spent;
        a dish that becomes overcrowded also loses cells. Readings are estimates.

        Args:
            dish: "A", "B" or "C".

        Returns:
            dict with "confluence_percent" (share of the floor covered),
            "attached_cells_estimate", "medium_ul", "medium_color" ("red", "orange"
            or "yellow") and "floating_cells" ("none", "few" or "many").
        """
        return self.sim.inspect(dish)

    def incubate(self, hours: float) -> None:
        """Put the dishes in the incubator for a while; the clock moves on.

        Only incubation moves the clock; robot work takes no culture time.

        Args:
            hours: how long, at most 24 per call.
        """
        self.sim.incubate(float(hours))

    def get_clock(self) -> float:
        """Get the culture clock in hours since the start of the episode."""
        return float(self.sim.clock)


CULTURE_PROMPT = """\
You control two I2RT YAM robot arms on a lab bench through the Python API below; this task uses the right arm only.
Task: keep a cancer cell line growing for one simulated week. The right gripper already holds a 1 mL pipette. The cells start in dish A; dishes B and C are empty, and there are reservoirs of fresh medium, trypsin and waste. Feed the cells, split them into more dishes before they overgrow, and finish the week with enough healthy cells. Call get_task() for the target. How fast the cells grow and how quickly they use their medium differ in every episode and are not told: inspect the dishes to find out. The positions of the dishes and reservoirs also differ.
An episode succeeds when, with the clock between 168 and 180 hours, the dishes together hold at least the target number of attached cells, at least 90 percent of the cells are alive, no dish starved or stayed overgrown for long, and every dish with cells is covered in medium."""


CULTURE_REFERENCE_POLICY = """
import numpy as np

task = get_task()
labware = get_labware()
offset = get_gripper_pose("right")["pos"] - get_pipette()["tip"]
TRAVEL = 0.06
CAPACITY = 1.0e6
VOLUME = 2000.0

def tip_to(point, speed=0.12):
    return move_to("right", np.asarray(point, dtype=float) + offset, 0.0, speed=speed)

def go_into(well, height):
    w = labware[well]
    tip_to([w["center"][0], w["center"][1], TRAVEL])
    tip_to([w["center"][0], w["center"][1], w["floor_z"] + height], speed=0.05)

def leave(well):
    w = labware[well]
    tip_to([w["center"][0], w["center"][1], TRAVEL], speed=0.08)

def volume(well):
    return get_labware()[well]["volume_ul"]

def transfer(source, target, amount):
    moved = 0.0
    while amount - moved > 1:
        go_into(source, 0.0005)                         # at the floor: draws down to the last drop
        got = aspirate(min(1000.0, amount - moved))["volume_ul"]
        leave(source)
        if got <= 0:
            break
        go_into(target, 0.006)
        dispense(got)
        leave(target)
        moved += got
    return moved

def empty(dish):
    transfer(dish, "waste", volume(dish))

def feed(dish):
    empty(dish)
    transfer("medium", dish, VOLUME)

def split(dish, targets):
    # Lift the cells, then share them out so every dish starts at the same density.
    empty(dish)
    transfer("trypsin", dish, 300)
    incubate(0.1)
    transfer("medium", dish, 900)                       # dilutes the trypsin to a quarter
    suspension = volume(dish)
    share = suspension / len(targets)
    for other in targets:
        if other != dish:
            transfer(dish, other, share)
    for d in targets:
        transfer("medium", d, VOLUME - volume(d))
    incubate(2.0)                                        # let them settle

dishes = ["A", "B", "C"]
in_use = ["A"]
for day in range(7):
    clock = get_clock()
    for d in list(in_use):
        look = inspect(d)
        free = [x for x in dishes if x not in in_use]
        if look["confluence_percent"] >= 40 and free and clock <= 100:
            split(d, [d] + free)                         # spread over every empty dish at once
            in_use += free
        elif look["medium_color"] != "red" or look["confluence_percent"] >= 35:
            feed(d)                                      # crowded dishes use a dish of medium a day
    incubate(min(24.0, 168.0 - get_clock()))
for d in in_use:
    print(d, inspect(d))
print("clock", get_clock())
"""
