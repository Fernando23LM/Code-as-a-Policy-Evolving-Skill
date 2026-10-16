"""Bench environments: the API a policy (hand-written or LLM-written) codes against."""
from __future__ import annotations

from dataclasses import dataclass

from collections import deque

import mujoco
import numpy as np

from . import judge
from .scene import (ARM_JOINTS, ARMS, BOAT_SIZE, CAP_LENGTH, CAP_THICK, GRIPPER_OPEN, HOME_Q, PAN_STIFFNESS, PAN_TOP,
                    RACK_BASE, TUBE_LENGTH, Layout, SnapLayout, WeighLayout, build_model)

CONTROL_DT = 0.02       # seconds of simulation per commanded waypoint
LINEAR_SPEED = 0.15     # m/s for Cartesian moves
JOINT_SPEED = 1.5       # rad/s for joint-space moves


@dataclass
class Task:
    tube: str
    dest_cell: tuple[int, int]

    @property
    def instruction(self) -> str:
        return f"Move {self.tube} from the source rack into cell {self.dest_cell} of the destination rack."


class TubeRackEnv:
    def __init__(self, seed: int | None = None, randomize: bool = True):
        self.randomize = randomize
        self._renderer = None
        self.on_tick = None  # callback for recorders and step budgets
        self.reset(seed)

    # ------------------------------------------------------------------ episode
    def reset(self, seed: int | None = None) -> dict:
        self._new_episode(np.random.default_rng(seed))
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.model = build_model(self.layout)
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        m = self.model
        self._qadr = {a: np.array([m.joint(f"{a}/{j}").qposadr[0] for j in ARM_JOINTS]) for a in ARMS}
        self._dadr = {a: np.array([m.joint(f"{a}/{j}").dofadr[0] for j in ARM_JOINTS]) for a in ARMS}
        self._act = {a: np.array([m.actuator(f"{a}/{j}").id for j in ARM_JOINTS]) for a in ARMS}
        self._grip = {a: m.actuator(f"{a}/gripper").id for a in ARMS}
        self._site = {a: m.site(f"{a}/pinch_site").id for a in ARMS}
        self._qrange = {a: np.array([m.joint(f"{a}/{j}").range for j in ARM_JOINTS]) for a in ARMS}

        for arm in ARMS:
            self.data.qpos[self._qadr[arm]] = HOME_Q
            self.data.ctrl[self._act[arm]] = HOME_Q
            self.data.ctrl[self._grip[arm]] = GRIPPER_OPEN
            self.data.qpos[m.joint(f"{arm}/left_finger").qposadr[0]] = GRIPPER_OPEN
            self.data.qpos[m.joint(f"{arm}/right_finger").qposadr[0]] = -GRIPPER_OPEN
        mujoco.mj_forward(m, self.data)
        self.step(0.5)  # let tubes settle into the rack
        return self.perceive()

    def _new_episode(self, rng: np.random.Generator) -> None:
        """Choose the layout and the task."""
        self.layout = Layout.sample(rng) if self.randomize else Layout()
        tubes, cells = list(self.layout.tubes), self.layout.dest.cells()
        self.task = Task(tubes[int(rng.integers(len(tubes)))], cells[int(rng.integers(len(cells)))])

    def step(self, seconds: float) -> None:
        """Advance the simulation; `on_tick`, if set, runs once per CONTROL_DT."""
        substeps = round(CONTROL_DT / self.model.opt.timestep)
        for _ in range(max(1, round(seconds / CONTROL_DT))):
            mujoco.mj_step(self.model, self.data, substeps)
            self._after_tick()
            if self.on_tick is not None:
                self.on_tick()

    def _after_tick(self) -> None:
        """Per-tick bookkeeping for subclasses (the balance filters its reading here)."""

    # --------------------------------------------------------------- perception
    def perceive(self) -> dict:
        """Ground-truth scene state. Stand-in for camera perception; swap out later."""
        d = self.data
        tubes = {}
        for name in self.layout.tubes:
            pos, mat = d.body(name).xpos.copy(), d.body(name).xmat.reshape(3, 3)
            tubes[name] = {"bottom": pos, "top": pos + mat[:, 2] * TUBE_LENGTH, "axis": mat[:, 2].copy()}
        racks = {}
        for rack in (self.layout.source, self.layout.dest):
            racks[rack.name] = {
                "yaw": rack.yaw,
                "cells": {c: np.array([*rack.cell_xy(*c), RACK_BASE]) for c in rack.cells()},
            }
        return {"tubes": tubes, "racks": racks, "ee": {a: self.ee_pose(a) for a in ARMS}}

    def ee_pose(self, arm: str) -> dict:
        """Pinch point between the finger pads, gripper yaw, and finger opening (m)."""
        site = self.data.site(self._site[arm])
        x_axis = site.xmat.reshape(3, 3)[:, 0]
        finger = self.data.qpos[self.model.joint(f"{arm}/left_finger").qposadr[0]]
        return {"pos": site.xpos.copy(), "yaw": float(np.arctan2(-x_axis[1], -x_axis[0])), "opening": 2 * float(finger)}

    def render(self, camera: str = "overhead", width: int = 640, height: int = 480) -> np.ndarray:
        """RGB image from 'overhead', 'front', 'left/wrist_cam' or 'right/wrist_cam'."""
        if self._renderer is None or (self._renderer.width, self._renderer.height) != (width, height):
            self._renderer = mujoco.Renderer(self.model, height, width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def camera_observation(self, camera: str = "overhead", width: int = 640, height: int = 480) -> dict:
        """RGB, metric depth, pinhole intrinsics and camera-to-world pose (OpenCV axes: z forward, y down)."""
        rgb = self.render(camera, width, height).copy()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera=camera)
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        cam = self.data.cam(camera)
        f = 0.5 * height / np.tan(np.radians(self.model.cam(camera).fovy[0]) / 2)
        pose = np.eye(4)
        pose[:3, :3] = cam.xmat.reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        pose[:3, 3] = cam.xpos
        return {
            "images": {"rgb": rgb, "depth": depth[..., None].astype(np.float32)},
            "intrinsics": np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]]),
            "pose_mat": pose,
        }

    # ------------------------------------------------------------------ control
    def move_to(self, arm: str, pos, yaw: float = 0.0, tilt: float = 0.0, speed: float = LINEAR_SPEED,
                tol: float = 0.001) -> bool:
        """Move the pinch point in a straight line to `pos`.

        `yaw` (rad, about world Z): 0 has the fingers closing along world Y, which is
        the arm's natural pose straight ahead. `tilt` (rad): 0 points the gripper
        straight down, positive leans it forward; straight down only works below
        about z = 0.15 to 0.19 depending on position. Returns False, without moving, if the
        target is unreachable.
        """
        start, goal = self.data.site(self._site[arm]).xpos.copy(), np.asarray(pos, dtype=float)
        n = max(2, int(np.ceil(np.linalg.norm(goal - start) / (speed * CONTROL_DT))))
        q = self.data.qpos[self._qadr[arm]].copy()
        path = []
        for t in np.linspace(0, 1, n + 1)[1:]:
            q, err = self._ik(arm, start + t * (goal - start), yaw, q, tilt)
            path.append(q)
        if err > 1e-3:
            return False
        # Re-time so no joint exceeds JOINT_SPEED (large yaw changes over short distances).
        prev = self.data.qpos[self._qadr[arm]].copy()
        for q in path:
            sub = max(1, int(np.ceil(np.abs(q - prev).max() / (JOINT_SPEED * CONTROL_DT))))
            for s in range(1, sub + 1):
                self.data.ctrl[self._act[arm]] = prev + (q - prev) * s / sub
                self.step(CONTROL_DT)
            prev = q
        return self._settle(arm, goal, path[-1], tol)

    def move_joints(self, arm: str, q) -> None:
        q, prev = np.asarray(q, dtype=float), self.data.ctrl[self._act[arm]].copy()
        n = max(1, int(np.ceil(np.abs(q - prev).max() / (JOINT_SPEED * CONTROL_DT))))
        for s in range(1, n + 1):
            self.data.ctrl[self._act[arm]] = prev + (q - prev) * s / n
            self.step(CONTROL_DT)
        self.step(0.3)

    def home(self, arm: str) -> None:
        self.move_joints(arm, HOME_Q)

    def set_gripper(self, arm: str, opening: float, wait: float = 0.6) -> None:
        """Command the distance between the finger pads (0 = closed, 0.082 = fully open)."""
        self.data.ctrl[self._grip[arm]] = float(np.clip(opening / 2, 0.0, GRIPPER_OPEN))
        self.step(wait)

    def _settle(self, arm: str, goal: np.ndarray, q_goal: np.ndarray, tol: float, timeout: float = 3.0) -> bool:
        """Integrate out the joint-friction offset the position servos leave behind."""
        for _ in range(int(timeout / CONTROL_DT)):
            self.step(CONTROL_DT)
            q_err = q_goal - self.data.qpos[self._qadr[arm]]
            if np.linalg.norm(self.data.site(self._site[arm]).xpos - goal) < tol and np.abs(q_err).max() < 0.006:
                return True
            if np.abs(self.data.qvel[self._dadr[arm]]).max() < 0.05:  # only correct once the arm is nearly at rest
                ctrl = self.data.ctrl[self._act[arm]] + 0.3 * q_err
                self.data.ctrl[self._act[arm]] = np.clip(ctrl, q_goal - 0.1, q_goal + 0.1)
        return False

    def _ik(self, arm: str, pos: np.ndarray, yaw: float, q0: np.ndarray, tilt: float = 0.0, iters: int = 200):
        """Damped least-squares IK. Returns (q, pose error)."""
        m, d = self.model, self._ik_data
        d.qpos[:] = self.data.qpos
        c, s, ct, st = np.cos(yaw), np.sin(yaw), np.cos(tilt), np.sin(tilt)
        # columns: long side, closing axis, approach (straight down when tilt = 0)
        target = np.array([[-c * ct, -s, c * st], [-s * ct, c, s * st], [-st, 0, -ct]])
        target_quat = np.empty(4)
        mujoco.mju_mat2Quat(target_quat, target.flatten())
        site, dofs, lim = self._site[arm], self._dadr[arm], self._qrange[arm]
        q, jacp, jacr = q0.copy(), np.zeros((3, m.nv)), np.zeros((3, m.nv))
        quat, neg, dq, rot_err = np.empty(4), np.empty(4), np.empty(4), np.empty(3)
        for _ in range(iters):
            d.qpos[self._qadr[arm]] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos_err = pos - d.site_xpos[site]
            mujoco.mju_mat2Quat(quat, d.site_xmat[site])
            mujoco.mju_negQuat(neg, quat)
            mujoco.mju_mulQuat(dq, target_quat, neg)
            mujoco.mju_quat2Vel(rot_err, dq, 1.0)
            err = np.concatenate([pos_err, rot_err])
            if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(rot_err) < 1e-3:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, site)
            J = np.vstack([jacp[:, dofs], jacr[:, dofs]])
            step = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            q = np.clip(q + np.clip(step, -0.3, 0.3), lim[:, 0], lim[:, 1])
        return q, float(np.linalg.norm(pos_err) + 0.05 * np.linalg.norm(rot_err))

    # -------------------------------------------------------------------- judge
    def evaluate(self) -> dict:
        """Score the episode. Not for use inside a policy."""
        return judge.evaluate(self)


@dataclass
class SnapCapTask:
    tube: str

    @property
    def instruction(self) -> str:
        return f"Open the snap cap of {self.tube} and leave it open."


class SnapCapEnv(TubeRackEnv):
    """Two snap-cap tubes clamped in one rack; open the lid of the named one."""

    def _new_episode(self, rng: np.random.Generator) -> None:
        self.layout = SnapLayout.sample(rng) if self.randomize else SnapLayout()
        tubes = list(self.layout.tubes)
        self.task = SnapCapTask(tubes[int(rng.integers(len(tubes)))])

    def cap_angle(self, tube: str) -> float:
        """Opening angle of a lid in radians, 0 when shut."""
        return float(self.data.qpos[self.model.joint(f"{tube}_hinge").qposadr[0]])

    def perceive(self) -> dict:
        """Ground-truth scene state. Stand-in for camera perception; swap out later."""
        d = self.data
        tubes = {}
        for name in self.layout.tubes:
            pos, mat = d.body(name).xpos.copy(), d.body(name).xmat.reshape(3, 3)
            lid = d.body(f"{name}_lid")
            lid_mat = lid.xmat.reshape(3, 3)
            tubes[name] = {
                "bottom": pos, "axis": mat[:, 2].copy(), "front": mat[:, 0].copy(), "hinge_axis": mat[:, 1].copy(),
                "hinge": lid.xpos.copy(), "angle": self.cap_angle(name),
                "cap_center": lid.xpos + lid_mat @ np.array([CAP_LENGTH / 2, 0, CAP_THICK / 2]),
                "yaw": self.layout.yaws[name],
            }
        rack = self.layout.rack
        racks = {rack.name: {"yaw": rack.yaw,
                             "cells": {c: np.array([*rack.cell_xy(*c), RACK_BASE]) for c in rack.cells()}}}
        return {"tubes": tubes, "racks": racks, "ee": {a: self.ee_pose(a) for a in ARMS}}

    def evaluate(self) -> dict:
        """Score the episode. Not for use inside a policy."""
        return judge.evaluate_snap_cap(self)


@dataclass
class WeighTask:
    target_g: float
    tolerance_g: float = 0.2

    @property
    def instruction(self) -> str:
        return (f"Weigh out {self.target_g:.1f} g of salt from the vial onto the weigh boat: the balance must read "
                f"within {self.tolerance_g:.2f} g of that.")


class WeighEnv(TubeRackEnv):
    """A vial of salt on the left arm's side, a balance on the right arm's side, a rack both can reach."""

    VIAL = "salt_vial"
    REPORT_GRIP = True      # the judge's note says how far the vial slid in a gripper
    SIM_BUDGET_S = 300.0    # two hand-offs and a slow pour take longer than the single-arm tasks
    WALL_BUDGET_S = 900
    READING_WINDOW = 25     # control ticks averaged by the display: half a second

    def _new_episode(self, rng: np.random.Generator) -> None:
        self.layout = WeighLayout.sample(rng) if self.randomize else WeighLayout()
        self.task = WeighTask(round(float(rng.integers(8, 17)) / 10, 1))     # 0.8 to 1.6 g of the 2.4 g in the vial
        self._deflection = deque(maxlen=self.READING_WINDOW)
        self._tare = None
        # How the vial shifts in each gripper while held: the judge reports it, policies do not see it.
        self._fingers = None
        self._held_at = {arm: None for arm in ARMS}
        self.grip_slip = {arm: 0.0 for arm in ARMS}      # metres the vial slid along the fingers, worst case
        self.grip_lean = {arm: 0.0 for arm in ARMS}      # degrees between vial and fingers, worst case

    def reset(self, seed: int | None = None) -> dict:
        super().reset(seed)
        self.step(0.5)                               # the pan settles under the boat's weight
        self._tare = float(np.mean(self._deflection))
        return self.perceive()

    def _after_tick(self) -> None:
        self._deflection.append(float(self.data.qpos[self.model.joint("balance_cell").qposadr[0]]))
        self._track_grip()

    def _track_grip(self) -> None:
        m, d = self.model, self.data
        if self._fingers is None:
            names = [m.body(b).name for b in range(m.nbody)]
            self._fingers = {arm: tuple({b for b, n in enumerate(names) if n.startswith(f"{arm}/{side}")}
                                        for side in ("lf", "rf")) for arm in ARMS}
            self._vial_id = m.body(self.VIAL).id
        touching = set()
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 == self._vial_id:
                touching.add(b2)
            elif b2 == self._vial_id:
                touching.add(b1)
        vial = d.body(self._vial_id)
        axis = vial.xmat.reshape(3, 3)[:, 2]
        for arm in ARMS:
            left, right = self._fingers[arm]
            if not (touching & left and touching & right):
                self._held_at[arm] = None
                continue
            site = d.site(self._site[arm])
            height = float(np.dot(site.xpos - vial.xpos, axis))          # where on the vial the pads are
            lean = float(np.degrees(np.arccos(np.clip(-np.dot(axis, site.xmat.reshape(3, 3)[:, 2]), -1, 1))))
            if self._held_at[arm] is None:
                self._held_at[arm] = height
            self.grip_slip[arm] = max(self.grip_slip[arm], height - self._held_at[arm])
            self.grip_lean[arm] = max(self.grip_lean[arm], lean)

    def balance_reading(self) -> float:
        """Grams on the pan since the tare at reset, as the display shows them: filtered, to 0.01 g."""
        if self._tare is None:                       # still settling inside reset
            return 0.0
        sag = self._tare - float(np.mean(self._deflection))
        return round(PAN_STIFFNESS * sag / 9.81 * 1000, 2)

    def salt_in_vial(self) -> tuple[int, int]:
        """(grains still in the vial's cup, grains in total)."""
        m, d = self.model, self.data
        vial = d.body(self.VIAL)
        frame = vial.xmat.reshape(3, 3)
        grains = [b for b in range(m.nbody) if m.body(b).name.startswith(f"{self.VIAL}_grain_")]
        local = [frame.T @ (d.xpos[b] - vial.xpos) for b in grains]
        return sum(1 for p in local if np.hypot(p[0], p[1]) < 0.0145 and 0.058 < p[2] < 0.082), len(grains)

    def balance_spread(self) -> float:
        """Peak-to-peak swing of the unfiltered reading over the display window, in grams."""
        sag = np.asarray(self._deflection)
        return float(PAN_STIFFNESS * (sag.max() - sag.min()) / 9.81 * 1000)

    def perceive(self) -> dict:
        """Ground-truth scene state. Stand-in for camera perception; swap out later."""
        d = self.data
        pos, mat = d.body(self.VIAL).xpos.copy(), d.body(self.VIAL).xmat.reshape(3, 3)
        tubes = {self.VIAL: {"bottom": pos, "top": pos + mat[:, 2] * TUBE_LENGTH, "axis": mat[:, 2].copy()}}
        racks = {}
        for rack in (self.layout.stock, self.layout.handover):
            racks[rack.name] = {"yaw": rack.yaw,
                                "cells": {c: np.array([*rack.cell_xy(*c), RACK_BASE]) for c in rack.cells()}}
        balance = {"reading_g": self.balance_reading(), "boat_size": BOAT_SIZE,
                   "boat_center": np.array([*self.layout.balance, PAN_TOP + float(d.qpos[self.model.joint("balance_cell").qposadr[0]])])}
        return {"tubes": tubes, "racks": racks, "balance": balance, "ee": {a: self.ee_pose(a) for a in ARMS}}

    def evaluate(self) -> dict:
        """Score the episode. Not for use inside a policy."""
        return judge.evaluate_weigh(self)


class WeighEnvPlain(WeighEnv):
    """The weighing task without the grip report, for comparing what that feedback is worth."""

    REPORT_GRIP = False


@dataclass
class HandoverTask:
    @property
    def instruction(self) -> str:
        return ("Bring the salt vial from the stock rack to the hand-over rack: finish with it standing upright in "
                "a cell of rack_handover, no gripper touching it, and no salt spilled.")


class WeighHandoverEnv(WeighEnv):
    """First half of the weighing task on its own: stand the vial in the rack both arms reach.

    It ends exactly where the pour sub-task starts, so the two learned scripts can be run one after the other.
    """

    def _new_episode(self, rng: np.random.Generator) -> None:
        super()._new_episode(rng)            # same layouts as the full task, seed for seed
        self.task = HandoverTask()

    def evaluate(self) -> dict:
        return judge.evaluate_handover(self)


class WeighPourEnv(WeighEnv):
    """Second half on its own: the vial already stands in the hand-over rack; weigh out the target mass."""

    def _new_episode(self, rng: np.random.Generator) -> None:
        super()._new_episode(rng)            # same racks, balance and target as the full task, seed for seed
        cells = self.layout.handover.cells()
        self.layout.in_handover = cells[int(rng.integers(len(cells)))]

