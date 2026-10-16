"""Success checks, one per task. Keep this file out of the agent's view."""
from __future__ import annotations

import numpy as np

from .scene import ARMS, RACK_BASE

XY_TOL = 0.006          # tube axis within 6 mm of the cell centre
SEAT_TOL = 0.006        # tube bottom within 6 mm of the rack base
TILT_TOL_DEG = 10.0
SPEED_TOL = 0.01        # m/s

CAP_OPEN_ANGLE = np.pi / 2   # a lid counts as open from 90 degrees
CAP_SHUT_ANGLE = 0.02        # rad; a latched lid sits at 0
CAP_SPEED_TOL = 0.05         # rad/s


def _in_cell(env, tube: str, rack, cell) -> bool:
    body = env.data.body(tube)
    xy_ok = np.linalg.norm(body.xpos[:2] - rack.cell_xy(*cell)) < XY_TOL
    return bool(xy_ok and abs(body.xpos[2] - RACK_BASE) < SEAT_TOL)


def _robot_touches(env, bodies: set[int]) -> bool:
    m, d = env.model, env.data
    robot_bodies = {b for b in range(m.nbody) if m.body(b).name.split("/")[0] in ARMS}
    return any(
        {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} & bodies
        and {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} & robot_bodies
        for c in d.contact[: d.ncon]
    )


def evaluate(env) -> dict:
    m, d, task = env.model, env.data, env.task
    body = d.body(task.tube)
    tilt = float(np.degrees(np.arccos(np.clip(body.xmat.reshape(3, 3)[2, 2], -1, 1))))
    speed = float(np.linalg.norm(body.cvel[3:]))

    held = _robot_touches(env, {m.body(task.tube).id})
    others_ok = all(
        _in_cell(env, name, env.layout.source, cell) for name, cell in env.layout.tubes.items() if name != task.tube
    )
    checks = {
        "in_target_cell": _in_cell(env, task.tube, env.layout.dest, task.dest_cell),
        "upright": tilt < TILT_TOL_DEG,
        "at_rest": speed < SPEED_TOL,
        "released": not held,
        "others_undisturbed": others_ok,
    }
    return {"success": all(checks.values()), "checks": checks, "tilt_deg": tilt}


def evaluate_snap_cap(env) -> dict:
    m, d, task = env.model, env.data, env.task
    angle = env.cap_angle(task.tube)
    speed = abs(float(d.qvel[m.joint(f"{task.tube}_hinge").dofadr[0]]))
    checks = {
        "cap_open": angle >= CAP_OPEN_ANGLE,
        "at_rest": speed < CAP_SPEED_TOL,
        "released": not _robot_touches(env, {m.body(task.tube).id, m.body(f"{task.tube}_lid").id}),
        "others_shut": all(env.cap_angle(name) < CAP_SHUT_ANGLE for name in env.layout.tubes if name != task.tube),
    }
    return {"success": all(checks.values()), "checks": checks, "angle_deg": float(np.degrees(angle))}


def evaluate_weigh(env) -> dict:
    """The balance reading decides; the pan must be at rest with nothing but salt on it."""
    m, d, task = env.model, env.data, env.task
    reading = env.balance_reading()
    pan = m.body("balance_pan").id
    others = {b for b in range(m.nbody) if m.body(b).name.split("/")[0] in ARMS} | {m.body(env.VIAL).id}
    touched = any({m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} == {pan, b} for c in d.contact[: d.ncon]
                  for b in ({m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} & others))
    checks = {
        "reading_in_tolerance": abs(reading - task.target_g) <= task.tolerance_g + 1e-9,
        "at_rest": env.balance_spread() < 0.05,
        "pan_clear": not touched,
    }
    # Partial credit, used to rank rounds that pass the same number of layouts and shown to the model:
    # 0.1 for getting the vial out of the stock rack, 0.1 for any salt on a clear pan, and 0.8 for how
    # close the reading is to the target. A vial or gripper on the pan earns nothing for the reading.
    start = env.layout.vial_start()
    moved = float(np.linalg.norm(d.body(env.VIAL).xpos[:2] - start) > 0.05)
    honest = not touched
    closeness = max(0.0, 1.0 - abs(reading - task.target_g) / task.target_g) if honest else 0.0
    progress = 1.0 if all(checks.values()) else 0.1 * moved + 0.1 * (honest and reading > 0.05) + 0.8 * closeness
    note = f"balance read {reading:.2f} g, target {task.target_g:.1f} g"
    vial = d.body(env.VIAL)
    if any({m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} == {pan, m.body(env.VIAL).id} for c in d.contact[: d.ncon]):
        note += "; the vial itself ended up on the pan and is being weighed"
    elif vial.xmat.reshape(3, 3)[2, 2] < 0.7:
        note += "; the vial ended up lying on its side"
    for arm in ARMS if env.REPORT_GRIP else ():
        if env.grip_slip[arm] >= 0.003:
            note += (f"; while the {arm} gripper held the vial it slid {env.grip_slip[arm] * 1000:.0f} mm down between "
                     f"the fingers and leaned up to {env.grip_lean[arm]:.0f} degrees away from them")
    return {"success": all(checks.values()), "checks": checks, "reading_g": reading,
            "progress": round(min(progress, 0.99) if not all(checks.values()) else 1.0, 3), "note": note}


def evaluate_handover(env) -> dict:
    """Hand-over sub-task: the vial stands in a cell of the hand-over rack, let go of, nothing spilled."""
    m, d = env.model, env.data
    vial = d.body(env.VIAL)
    tilt = float(np.degrees(np.arccos(np.clip(vial.xmat.reshape(3, 3)[2, 2], -1, 1))))
    rack = env.layout.handover
    seated = any(_in_cell(env, env.VIAL, rack, cell) for cell in rack.cells())
    kept, total = env.salt_in_vial()
    checks = {
        "in_handover_cell": seated,
        "upright": tilt < TILT_TOL_DEG,
        "released": not _robot_touches(env, {vial.id}),
        "no_salt_spilled": kept == total,
        "at_rest": float(np.linalg.norm(vial.cvel[3:])) < SPEED_TOL,
    }
    success = all(checks.values())
    # Partial credit: out of the stock rack, upright in the strip both arms reach, seated in a cell, salt kept.
    moved = float(np.linalg.norm(vial.xpos[:2] - env.layout.vial_start()) > 0.05)
    across = float(abs(vial.xpos[1]) < 0.08 and tilt < TILT_TOL_DEG)
    progress = 1.0 if success else min(0.99, 0.25 * moved + 0.25 * across + 0.3 * seated + 0.2 * kept / total * across)
    note = (f"vial at {np.round(vial.xpos, 3).tolist()}, leaning {tilt:.0f} degrees, {kept} of {total} grains still in it, "
            f"{'seated in' if seated else 'not seated in'} a hand-over cell")
    return {"success": success, "checks": checks, "progress": round(progress, 3), "note": note}
