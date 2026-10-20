"""Imperfections of a real bench, to close part of the sim-to-real gap.

A `Realism` is a set of error sizes; each episode draws its own errors from it (`RealismEpisode`):

- sensing: every position the policy reads (vial, rack, lid, labware, pipette tip, balance boat) carries a
  fixed per-session camera calibration error plus fresh jitter on every reading; angles and volumes too;
  the balance display carries noise.
- the arm: it lands away from where it is sent by a fixed per-session calibration error plus a little
  repeatability error, and reports its own position in its own (miscalibrated) frame.
- pipetting: delivered volumes are off by a per-session bias plus per-stroke scatter.
- physics: friction, the masses of loose objects, joint stiffness and grip force vary from session to session.

The defaults are assumptions, not measurements. Replace them with values measured on the real YAM arms,
cameras and pipette (see README). Every task has a variant named `<task>_real` that uses them.
The judge always scores the true state, never the noisy readings.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

ARMS = ("left", "right")


@dataclass(frozen=True)
class Realism:
    camera_bias_mm: float = 1.0     # per-session error of every camera position, standard deviation per axis
    camera_noise_mm: float = 0.5    # fresh error on every reading
    angle_noise_rad: float = 0.01   # on reported yaws and lid angles
    volume_read_cv: float = 0.02    # relative error on reported liquid volumes
    balance_noise_g: float = 0.005  # on each balance reading
    arm_bias_mm: float = 1.0        # per-session error between where an arm is sent and where it goes
    arm_repeat_mm: float = 0.2      # fresh error on every move
    yaw_bias_rad: float = 0.005     # per-session wrist rotation error
    pipette_bias: float = 0.01      # per-session relative volume error
    pipette_cv: float = 0.005       # per-stroke relative volume scatter
    friction_spread: float = 0.2    # session friction factor, uniform in 1 +/- this
    mass_spread: float = 0.1        # loose objects' mass factor
    gain_spread: float = 0.1        # joint stiffness factor
    grip_spread: float = 0.15       # grip force factor

    def episode(self, seed: int | None, model: mujoco.MjModel) -> "RealismEpisode":
        rng = np.random.default_rng(None if seed is None else [int(seed), 20261017])
        ep = RealismEpisode(self, rng)
        ep.randomize_physics(model)
        return ep


class RealismEpisode:
    def __init__(self, cfg: Realism, rng: np.random.Generator):
        mm = 0.001
        self.cfg, self.rng = cfg, rng
        self.camera_bias = rng.normal(0, cfg.camera_bias_mm * mm, 3)
        self.arm_bias = {arm: rng.normal(0, cfg.arm_bias_mm * mm, 3) for arm in ARMS}
        self.yaw_bias = {arm: float(rng.normal(0, cfg.yaw_bias_rad)) for arm in ARMS}
        self.pipette_bias = float(rng.normal(0, cfg.pipette_bias))

    # ---------------------------------------------------------------- physics
    def randomize_physics(self, m: mujoco.MjModel) -> None:
        cfg, rng = self.cfg, self.rng
        session = rng.uniform(1 - cfg.friction_spread, 1 + cfg.friction_spread)
        m.geom_friction[:, 0] *= session * rng.normal(1.0, 0.05, m.ngeom).clip(0.85, 1.15)
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                b = m.jnt_bodyid[j]
                f = rng.uniform(1 - cfg.mass_spread, 1 + cfg.mass_spread)
                m.body_mass[b] *= f
                m.body_inertia[b] *= f
        for a in range(m.nu):
            name = m.actuator(a).name
            if name.split("/")[0] not in ARMS:
                continue
            f = rng.uniform(1 - (cfg.grip_spread if name.endswith("gripper") else cfg.gain_spread),
                            1 + (cfg.grip_spread if name.endswith("gripper") else cfg.gain_spread))
            m.actuator_gainprm[a][0] *= f
            m.actuator_biasprm[a][1] *= f
            if name.endswith("gripper"):
                m.actuator_forcerange[a] *= f

    # ---------------------------------------------------------------- the arm
    def command(self, arm: str, pos: np.ndarray, yaw: float) -> tuple[np.ndarray, float]:
        """Where the arm really goes when sent to `pos`, `yaw` in its own frame."""
        return (np.asarray(pos, dtype=float) + self.arm_bias[arm] + self.rng.normal(0, self.cfg.arm_repeat_mm * 0.001, 3),
                float(yaw) + self.yaw_bias[arm])

    def reported_pose(self, arm: str, pose: dict) -> dict:
        """The arm's report of its own pose: the true pose seen through its calibration error."""
        out = dict(pose)
        out["pos"] = pose["pos"] - self.arm_bias[arm] + self.rng.normal(0, self.cfg.arm_repeat_mm * 0.0005, 3)
        out["yaw"] = float(pose["yaw"]) - self.yaw_bias[arm]
        return out

    # ---------------------------------------------------------------- sensing
    def observe(self, obs, key: str = ""):
        """A copy of a perception dict with sensing errors applied to everything the policy reads."""
        cfg, rng = self.cfg, self.rng
        if isinstance(obs, dict):
            return {k: self.observe(v, k if isinstance(k, str) else key) for k, v in obs.items()}
        if isinstance(obs, np.ndarray) and obs.shape == (3,):
            if abs(np.linalg.norm(obs) - 1.0) < 1e-6:                 # a direction: tilt it slightly
                v = obs + rng.normal(0, cfg.angle_noise_rad, 3)
                return v / np.linalg.norm(v)
            return obs + self.camera_bias + rng.normal(0, cfg.camera_noise_mm * 0.001, 3)
        if isinstance(obs, float) and key == "floor_z":                 # a height seen by the cameras
            return obs + float(self.camera_bias[2] + rng.normal(0, cfg.camera_noise_mm * 0.001))
        if isinstance(obs, float) and key in ("yaw", "angle"):
            return obs + float(rng.normal(0, cfg.angle_noise_rad))
        if isinstance(obs, float) and key == "volume_ul":
            return round(obs * (1 + float(rng.normal(0, cfg.volume_read_cv))), 1)
        if isinstance(obs, float) and key == "reading_g":
            return round(obs + float(rng.normal(0, cfg.balance_noise_g)), 2)
        return obs

    def pipette_factor(self) -> float:
        """Delivered / requested volume for one stroke."""
        return max(0.0, 1.0 + self.pipette_bias + float(self.rng.normal(0, self.cfg.pipette_cv)))
