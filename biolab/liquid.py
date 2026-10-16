"""Liquid handling without fluid simulation: wells hold a volume and a concentration, a pipette moves them.

A well is any open container fixed to the bench. Liquid in it is well mixed and fills it from the floor up,
so its surface height follows from the volume and the opening's area. The pipette is a free body held in a
gripper; its tip is the body origin. Aspirating draws liquid only when the tip is in a well and below the
surface, otherwise it draws air; dispensing outside a well spills. There is no residue left in the tip.
Shared by the protein assay and the cell-culture tasks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TIP_RADIUS = 0.0015
SUBMERGED = 0.0005          # the tip must be at least this far below the surface to draw liquid
FLOOR_DRAW = 0.0008         # ...unless it is right at the floor, where it can draw the last drops


@dataclass
class Well:
    name: str
    center: np.ndarray       # (2,) xy of the opening's centre, world frame
    floor: float             # z of the inner floor
    half: float              # half the side of the square opening
    depth: float             # floor to rim
    volume: float = 0.0      # microlitres
    conc: float = 0.0        # any amount per microlitre unit the task likes (mg/mL for protein)
    extra: dict = field(default_factory=dict)   # task-specific contents, e.g. cells
    yaw: float = 0.0         # rotation of the opening about world Z

    @property
    def area_mm2(self) -> float:
        return (2 * self.half * 1000) ** 2

    @property
    def capacity(self) -> float:
        return self.area_mm2 * self.depth * 1000

    @property
    def surface(self) -> float:
        """z of the liquid surface (the floor when empty)."""
        return self.floor + self.volume / self.area_mm2 / 1000

    def holds(self, tip: np.ndarray) -> bool:
        """Is the tip inside the opening and below the rim?"""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        dx, dy = tip[:2] - self.center
        local = np.array([c * dx + s * dy, -s * dx + c * dy])      # into the well's own frame
        inside = np.all(np.abs(local) <= self.half - TIP_RADIUS)
        return bool(inside and self.floor - 0.002 < tip[2] < self.floor + self.depth)

    def add(self, volume: float, conc: float, extra: dict | None = None) -> float:
        """Pour in liquid carrying `extra` totals (e.g. cells); returns what did not fit (overflow)."""
        fits = max(0.0, min(volume, self.capacity - self.volume))
        if fits > 0:
            total = self.volume + fits
            self.conc = (self.conc * self.volume + conc * fits) / total
            self.volume = total
            for key, value in (extra or {}).items():        # extensive quantities, e.g. cell counts
                self.extra[key] = self.extra.get(key, 0.0) + value * fits / volume
        return volume - fits

    def take(self, volume: float) -> tuple[float, float, dict]:
        """Draw liquid out; returns (volume, concentration, share of the extras)."""
        taken, conc = min(volume, self.volume), self.conc
        share = taken / self.volume if self.volume > 0 else 0.0
        extras = {key: value * share for key, value in self.extra.items()}
        for key in self.extra:
            self.extra[key] -= extras[key]
        self.volume -= taken
        if self.volume <= 1e-9:
            self.volume, self.conc = 0.0, 0.0
        return taken, conc if taken > 0 else 0.0, extras


@dataclass
class Pipette:
    capacity: float = 200.0    # microlitres
    volume: float = 0.0
    conc: float = 0.0
    extra: dict = field(default_factory=dict)
    spilled: float = 0.0       # microlitres dispensed outside any well, or lost to overflow
    log: list = field(default_factory=list)

    def aspirate(self, wells: list[Well], tip: np.ndarray, volume: float) -> dict:
        well = next((w for w in wells if w.holds(tip)), None)
        room = self.capacity - self.volume
        want = max(0.0, min(volume, room))
        reaches = tip[2] <= well.surface - SUBMERGED or tip[2] - well.floor < FLOOR_DRAW if well else False
        if well is None or not reaches or well.volume <= 0:
            result = {"volume_ul": 0.0, "from": well.name if well else None, "note": "drew air"}
        else:
            taken, conc, extras = well.take(want)
            if taken > 0:
                total = self.volume + taken
                self.conc = (self.conc * self.volume + conc * taken) / total
                self.volume = total
                for key, value in extras.items():
                    self.extra[key] = self.extra.get(key, 0.0) + value
            result = {"volume_ul": round(taken, 3), "from": well.name}
        self.log.append(("aspirate", round(volume, 3), result))
        return result

    def dispense(self, wells: list[Well], tip: np.ndarray, volume: float) -> dict:
        out = max(0.0, min(volume, self.volume))
        share = out / self.volume if self.volume > 0 else 0.0
        extras = {key: value * share for key, value in self.extra.items()}
        for key in self.extra:
            self.extra[key] -= extras[key]
        conc = self.conc
        self.volume -= out
        if self.volume <= 1e-9:
            self.volume, self.conc = 0.0, 0.0
        well = next((w for w in wells if w.holds(tip)), None)
        if well is None:
            self.spilled += out
            result = {"volume_ul": 0.0, "into": None, "note": f"{out:.1f} uL spilled outside any well"}
        else:
            overflow = well.add(out, conc, extras)
            self.spilled += overflow
            result = {"volume_ul": round(out - overflow, 3), "into": well.name}
            if overflow > 0:
                result["note"] = f"{overflow:.1f} uL overflowed"
        self.log.append(("dispense", round(volume, 3), result))
        return result
