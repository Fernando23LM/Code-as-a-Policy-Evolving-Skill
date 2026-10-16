"""Bench scene: two I2RT YAM arms, a source tube rack and a destination rack.

Conventions (same as Documents/BiomedicalBenchmarks): metres, Z up, table top
is the plane z = 0, arms at (-0.20, +/-0.28, 0) facing +X.

The YAM can only point its gripper straight down within about 0.10-0.35 m of
its base and below z = 0.16 (wrist pitch limit), so both racks sit inside that
envelope of the right arm and the vials are short enough to carry over each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

YAM_XML = Path(__file__).resolve().parents[1] / "third_party/mujoco_menagerie/i2rt_yam/yam.xml"

ARMS = ("left", "right")
ARM_BASE = {"left": (-0.20, 0.28, 0.0), "right": (-0.20, -0.28, 0.0)}
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
HOME_Q = (0.0, 1.047, 1.047, 0.0, 0.0, 0.0)
GRIPPER_OPEN = 0.041  # finger actuator ctrl, metres of travel per finger

TUBE_RADIUS = 0.014
TUBE_LENGTH = 0.075
TUBE_MASS = 0.03

RACK_HOLE = 0.032    # square opening, 2 mm clearance each side of a tube
RACK_WALL = 0.005
RACK_PITCH = RACK_HOLE + RACK_WALL
RACK_HEIGHT = 0.035
RACK_BASE = 0.006

# Snap-cap tube: a body clamped in the rack and a flat lid on a hinge along its back bottom edge.
SNAP_BODY_RADIUS = 0.011
CAP_LENGTH = 0.030       # hinge to front edge
CAP_WIDTH = 0.028        # along the hinge
CAP_THICK = 0.010
CAP_MAX_ANGLE = 2.2      # rad, where the open lid hits its stop
LATCH_TORQUE = 0.09      # N m holding a shut lid: 3 N at the front edge, 6 N at the middle
LATCH_ANGLE = 0.12       # rad, the latch lets go past this

# Weighing task: an open vial of coarse salt, and a balance with a weigh boat fixed on its pan.
CUP_DEPTH = 0.015        # the salt sits in a cup in the top of the vial; the rest of the vial is solid
CUP_RADIUS = 0.0072      # at the cup's floor
CUP_FLARE = np.radians(20)
GRAIN_RADIUS = 0.0018    # the largest grain; sizes vary down to GRAIN_RADIUS_MIN
GRAIN_RADIUS_MIN = 0.0014
GRAIN_MASS = 5e-5        # 0.05 g for the largest grain, 0.024 g for the smallest
GRAIN_LAYERS = 4         # 13 + 13 + 13 + 29 grains: 68 grains, about 2.4 g, a nearly full cup
GRAIN_SOLREF = [0.005, 1]  # default contacts are soft enough for 3.6 mm grains to sink into each other
BOAT_SIZE = 0.064
BOAT_RIM = 0.012
PAN_TOP = 0.028          # height of the pan's upper face above the table
PAN_STIFFNESS = 400.0    # N/m, the load cell

TUBE_COLORS = ((0.85, 0.2, 0.2, 1), (0.2, 0.45, 0.85, 1), (0.95, 0.75, 0.1, 1), (0.25, 0.7, 0.35, 1))


@dataclass
class Rack:
    name: str
    xy: tuple[float, float]
    nx: int = 4
    ny: int = 2
    yaw: float = 0.0

    def cell_xy(self, ix: int, iy: int) -> np.ndarray:
        """World xy of the centre of cell (ix, iy)."""
        local = np.array([(ix - (self.nx - 1) / 2) * RACK_PITCH, (iy - (self.ny - 1) / 2) * RACK_PITCH])
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array(self.xy) + np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])

    def cells(self) -> list[tuple[int, int]]:
        return [(ix, iy) for iy in range(self.ny) for ix in range(self.nx)]


@dataclass
class Layout:
    """Everything that varies between episodes."""
    source: Rack = field(default_factory=lambda: Rack("rack_source", (0.02, -0.37)))
    dest: Rack = field(default_factory=lambda: Rack("rack_dest", (0.02, -0.19)))
    # tube name -> cell of the source rack it starts in
    tubes: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"tube_0": (0, 0), "tube_1": (3, 1)})

    @staticmethod
    def sample(rng: np.random.Generator, n_tubes: int = 2) -> "Layout":
        def jitter(xy, yaw_max=0.35):
            return (xy[0] + rng.uniform(-0.02, 0.02), xy[1] + rng.uniform(-0.02, 0.02)), rng.uniform(-yaw_max, yaw_max)

        (sxy, syaw), (dxy, dyaw) = jitter((0.02, -0.37)), jitter((0.02, -0.19))
        source, dest = Rack("rack_source", sxy, yaw=syaw), Rack("rack_dest", dxy, yaw=dyaw)
        return Layout(source, dest, _pick_cells(rng, source, n_tubes))


@dataclass
class SnapLayout:
    """Snap-cap task: one rack of clamped tubes, each turned its own way."""
    rack: Rack = field(default_factory=lambda: Rack("rack", (0.02, -0.28)))
    tubes: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"tube_0": (0, 0), "tube_1": (3, 1)})
    # tube name -> world yaw of the direction from the hinge to the lid's front edge
    yaws: dict[str, float] = field(default_factory=lambda: {"tube_0": 0.0, "tube_1": 0.0})

    @staticmethod
    def sample(rng: np.random.Generator, n_tubes: int = 2) -> "SnapLayout":
        xy = (0.02 + rng.uniform(-0.03, 0.03), -0.28 + rng.uniform(-0.03, 0.03))
        rack = Rack("rack", xy, yaw=rng.uniform(-0.35, 0.35))
        tubes = _pick_cells(rng, rack, n_tubes)
        return SnapLayout(rack, tubes, {name: float(rng.uniform(-np.pi, np.pi)) for name in tubes})


@dataclass
class WeighLayout:
    """Weighing task: the salt starts on the left arm's side, the balance is on the right arm's side."""
    stock: Rack = field(default_factory=lambda: Rack("rack_stock", (0.02, 0.37)))
    handover: Rack = field(default_factory=lambda: Rack("rack_handover", (-0.08, 0.0), nx=2, ny=1))
    balance: tuple[float, float] = (0.08, -0.28)
    cell: tuple[int, int] = (0, 0)       # where the salt vial starts in the stock rack
    in_handover: tuple[int, int] | None = None   # pour-only task: it starts in this cell of the hand-over rack

    def vial_start(self) -> np.ndarray:
        if self.in_handover is not None:
            return self.handover.cell_xy(*self.in_handover)
        return self.stock.cell_xy(*self.cell)

    @staticmethod
    def sample(rng: np.random.Generator) -> "WeighLayout":
        stock = Rack("rack_stock", (0.02 + rng.uniform(-0.02, 0.02), 0.37 + rng.uniform(-0.02, 0.02)),
                     yaw=rng.uniform(-0.35, 0.35))
        # Both arms reach only a narrow strip between them, so the handover rack hardly moves.
        handover = Rack("rack_handover", (-0.08 + rng.uniform(-0.008, 0.008), rng.uniform(-0.015, 0.015)),
                        nx=2, ny=1, yaw=rng.uniform(-0.3, 0.3))
        balance = (0.08 + rng.uniform(-0.02, 0.02), -0.28 + rng.uniform(-0.02, 0.02))
        cells = stock.cells()
        return WeighLayout(stock, handover, balance, cells[int(rng.integers(len(cells)))])


def _pick_cells(rng: np.random.Generator, rack: Rack, n_tubes: int) -> dict[str, tuple[int, int]]:
    # No two tubes in touching cells, diagonals included: the fingers need a free cell on each
    # side and the finger struts reach over the diagonal cells.
    cells = rack.cells()
    while True:
        picks = [cells[int(j)] for j in rng.choice(len(cells), size=n_tubes, replace=False)]
        if all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1 for i, a in enumerate(picks) for b in picks[:i]):
            return {f"tube_{i}": c for i, c in enumerate(picks)}


def _add_rack(spec: mujoco.MjSpec, rack: Rack, rgba) -> None:
    body = spec.worldbody.add_body(name=rack.name, pos=[*rack.xy, 0], quat=_yaw_quat(rack.yaw))
    lx, ly = rack.nx * RACK_PITCH + RACK_WALL, rack.ny * RACK_PITCH + RACK_WALL
    box = dict(type=mujoco.mjtGeom.mjGEOM_BOX, rgba=rgba)
    body.add_geom(name=f"{rack.name}_base", size=[lx / 2, ly / 2, RACK_BASE / 2], pos=[0, 0, RACK_BASE / 2], **box)
    zc, hz = (RACK_BASE + RACK_HEIGHT) / 2, (RACK_HEIGHT - RACK_BASE) / 2
    for i in range(rack.nx + 1):
        body.add_geom(size=[RACK_WALL / 2, ly / 2, hz], pos=[-lx / 2 + RACK_WALL / 2 + i * RACK_PITCH, 0, zc], **box)
    for j in range(rack.ny + 1):
        body.add_geom(size=[lx / 2, RACK_WALL / 2, hz], pos=[0, -ly / 2 + RACK_WALL / 2 + j * RACK_PITCH, zc], **box)


def _add_tube(spec: mujoco.MjSpec, name: str, xy, rgba) -> None:
    """Capped sample vial (28 mm x 75 mm). The body origin is the bottom centre."""
    body = spec.worldbody.add_body(name=name, pos=[xy[0], xy[1], RACK_BASE + 0.001])
    body.add_freejoint(name=f"{name}_free")
    cap = 0.012
    body_len = TUBE_LENGTH - cap
    cyl = dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER, friction=[1.5, 0.01, 0.0005], condim=4)
    # The vial stands on a thin box: a cylinder face resting on a box rocks forever in MuJoCo 3.5,
    # the version CaP-X locks.
    foot, lift = 0.0095, 0.001
    body.add_geom(name=f"{name}_foot", type=mujoco.mjtGeom.mjGEOM_BOX, size=[foot, foot, lift], pos=[0, 0, lift],
                  rgba=[0.9, 0.9, 0.95, 0.6], mass=0.001, friction=[1.0, 0.01, 0.0005])
    body.add_geom(name=f"{name}_body", size=[TUBE_RADIUS, (body_len - lift) / 2, 0], pos=[0, 0, (body_len + lift) / 2],
                  rgba=[0.9, 0.9, 0.95, 0.6], mass=TUBE_MASS * 0.85, **cyl)
    body.add_geom(name=f"{name}_cap", size=[TUBE_RADIUS, cap / 2, 0], pos=[0, 0, body_len + cap / 2],
                  rgba=rgba, mass=TUBE_MASS * 0.15, **cyl)


def _add_snap_tube(spec: mujoco.MjSpec, name: str, xy, yaw: float, rgba) -> None:
    """Snap-cap tube, same 75 mm height as the vial. Body origin is the bottom centre, +x the lid's front.

    The body has no joint: the rack clamps it, so lifting the lid does not lift the tube.
    """
    body_len = TUBE_LENGTH - CAP_THICK
    body = spec.worldbody.add_body(name=name, pos=[xy[0], xy[1], RACK_BASE], quat=_yaw_quat(yaw))
    body.add_geom(name=f"{name}_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[SNAP_BODY_RADIUS, body_len / 2, 0],
                  pos=[0, 0, body_len / 2], rgba=[0.9, 0.9, 0.95, 0.6])
    lid = body.add_body(name=f"{name}_lid", pos=[-CAP_LENGTH / 2, 0, body_len])
    lid.add_joint(name=f"{name}_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, -1, 0],
                  range=[0, np.degrees(CAP_MAX_ANGLE)], damping=0.002, frictionloss=0.002, armature=5e-5,
                  solref_limit=[0.004, 1], solimp_limit=[0.99, 0.999, 0.0001, 0.5, 2])  # stiff stop for a light lid
    spec.add_exclude(bodyname1=name, bodyname2=f"{name}_lid")  # the hinge stop seats the lid, not the rim
    lid.add_geom(name=f"{name}_cap", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CAP_LENGTH / 2, CAP_WIDTH / 2, CAP_THICK / 2],
                 pos=[CAP_LENGTH / 2, 0, CAP_THICK / 2], rgba=rgba, mass=0.004, friction=[1.5, 0.01, 0.0005], condim=4)
    # Snap latch: a closing torque that fades to zero at LATCH_ANGLE and is clamped to zero beyond it,
    # so a lid that has let go turns freely and stays where it is left (joint friction beats its weight).
    latch = spec.add_actuator(name=f"{name}_latch", target=f"{name}_hinge", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    latch.gaintype, latch.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
    latch.gainprm[0] = 0
    latch.biasprm[:3] = [-LATCH_TORQUE, LATCH_TORQUE / LATCH_ANGLE, 0]
    latch.forcelimited, latch.forcerange = True, [-LATCH_TORQUE, 0]


def _add_salt_vial(spec: mujoco.MjSpec, name: str, xy) -> list[str]:
    """Open vial with the outside of the task-1 vial (28 mm x 75 mm) and a cup of salt grains in its top.

    Returns the names of the grain bodies. The outside is one solid cylinder for the grippers and racks;
    the cup is a ring of boxes in a child body, and the grains are excluded from touching the outside.
    """
    body = spec.worldbody.add_body(name=name, pos=[xy[0], xy[1], RACK_BASE + 0.001])
    body.add_freejoint(name=f"{name}_free")
    foot, lift = 0.0095, 0.001
    body.add_geom(name=f"{name}_foot", type=mujoco.mjtGeom.mjGEOM_BOX, size=[foot, foot, lift], pos=[0, 0, lift],
                  rgba=[0.9, 0.9, 0.95, 0.6], mass=0.001, friction=[1.0, 0.01, 0.0005])
    body.add_geom(name=f"{name}_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[TUBE_RADIUS, (TUBE_LENGTH - lift) / 2, 0], pos=[0, 0, (TUBE_LENGTH + lift) / 2],
                  rgba=[0.75, 0.85, 0.95, 0.45], mass=TUBE_MASS, friction=[1.5, 0.01, 0.0005], condim=4)
    cup = body.add_body(name=f"{name}_cup")
    floor_z = TUBE_LENGTH - CUP_DEPTH
    # The cup touches only grains (collision bit 2): its flared rim would otherwise foul the finger plates.
    wall = dict(type=mujoco.mjtGeom.mjGEOM_BOX, rgba=[0.75, 0.85, 0.95, 0.45], mass=1e-4, friction=[0.25, 0.005, 0.0001],
                solref=GRAIN_SOLREF, contype=2, conaffinity=0)
    cup.add_geom(name=f"{name}_cup_floor", size=[0.011, 0.011, 0.002], pos=[0, 0, floor_z - 0.002], **wall)
    # The wall flares outwards like a funnel, so the salt pours well before the vial is horizontal.
    sides, thick, slope = 12, 0.0012, np.tan(CUP_FLARE)
    top_radius = CUP_RADIUS + CUP_DEPTH * slope
    flare_quat = np.array([np.cos(CUP_FLARE / 2), 0, np.sin(CUP_FLARE / 2), 0])
    for i in range(sides):
        a = 2 * np.pi * i / sides
        r = CUP_RADIUS + CUP_DEPTH / 2 * slope + thick / 2
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, np.array(_yaw_quat(a)), flare_quat)
        cup.add_geom(size=[thick / 2, (top_radius + thick) * np.tan(np.pi / sides), CUP_DEPTH / 2 / np.cos(CUP_FLARE)],
                     pos=[r * np.cos(a), r * np.sin(a), floor_z + CUP_DEPTH / 2], quat=quat, **wall)
    grains, pitch = [], 2 * GRAIN_RADIUS + 0.00005
    rings = ((0.0021, 3), (0.0058, 10), (0.0095, 16))
    # Mixed sizes: equal spheres lock into layers and leave the cup in avalanches of ten at a time.
    sizes = np.random.default_rng(0).uniform(GRAIN_RADIUS_MIN, GRAIN_RADIUS, 200)
    for layer in range(GRAIN_LAYERS):
        z = GRAIN_RADIUS + 0.0002 + layer * pitch
        fits = CUP_RADIUS + z * slope - GRAIN_RADIUS / np.cos(CUP_FLARE) - 0.0001    # widest ring this layer holds
        for rad, n in rings:
            if rad > fits:
                break
            for k in range(n):
                grain = f"{name}_grain_{len(grains)}"
                turn = 2 * np.pi * k / n + rad * 40 + layer
                g = spec.worldbody.add_body(name=grain, pos=[xy[0] + rad * np.cos(turn), xy[1] + rad * np.sin(turn),
                                                             RACK_BASE + 0.001 + floor_z + z])
                g.add_freejoint()
                radius = float(sizes[len(grains)])
                g.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[radius, 0, 0],
                           mass=GRAIN_MASS * (radius / GRAIN_RADIUS) ** 3,
                           rgba=[1, 1, 1, 1], friction=[0.25, 0.005, 0.0001], solref=GRAIN_SOLREF, conaffinity=3)
                spec.add_exclude(bodyname1=name, bodyname2=grain)
                grains.append(grain)
    return grains


def _add_balance(spec: mujoco.MjSpec, xy) -> None:
    """Balance: a pan on a stiff vertical spring (the load cell) with a weigh boat fixed on it."""
    box = mujoco.mjtGeom.mjGEOM_BOX
    # Narrower than the pan, so a stray grain cannot lodge beside the pan and prop it up.
    spec.worldbody.add_geom(name="balance_base", type=box, size=[0.035, 0.035, 0.011], pos=[xy[0], xy[1], 0.011],
                            rgba=[0.25, 0.27, 0.3, 1])
    pan = spec.worldbody.add_body(name="balance_pan", pos=[xy[0], xy[1], PAN_TOP])
    pan.add_joint(name="balance_cell", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], stiffness=PAN_STIFFNESS,
                  damping=10.0, range=[-0.003, 0.003])
    pan.add_geom(name="balance_plate", type=box, size=[0.045, 0.045, 0.002], pos=[0, 0, -0.002], mass=0.05,
                 rgba=[0.75, 0.76, 0.78, 1])
    # Rolling friction on the boat so grains come to rest instead of rolling about.
    boat = dict(type=box, rgba=[0.97, 0.97, 0.94, 1], mass=0.0005, friction=[0.6, 0.005, 0.0004], condim=6,
                solref=GRAIN_SOLREF)
    half, t = BOAT_SIZE / 2, 0.001
    pan.add_geom(name="boat_floor", size=[half, half, t], pos=[0, 0, t], **boat)
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        pan.add_geom(size=[t if sx else half, t if sy else half, BOAT_RIM / 2],
                     pos=[sx * (half - t), sy * (half - t), BOAT_RIM / 2], **boat)


def _yaw_quat(yaw: float) -> list[float]:
    return [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def build_spec(layout: Layout | SnapLayout | WeighLayout | None = None) -> mujoco.MjSpec:
    layout = layout or Layout()
    spec = mujoco.MjSpec()
    spec.modelname = "biolab_tube_rack"
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10  # stiffer friction so grasped tubes do not creep
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960

    world = spec.worldbody
    world.add_light(pos=[0.2, 0, 1.5], dir=[0, 0, -1], diffuse=[0.45, 0.45, 0.45])
    world.add_light(pos=[-0.5, 0.8, 1.2], dir=[0.4, -0.6, -0.7], diffuse=[0.25, 0.25, 0.25])
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05], pos=[0, 0, -0.75],
                   rgba=[0.35, 0.37, 0.4, 1])
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.45, 0.60, 0.02], pos=[0.10, 0, -0.02],
                   rgba=[0.62, 0.6, 0.56, 1])

    for arm in ARMS:
        yam = mujoco.MjSpec.from_file(str(YAM_XML))
        for element in [*yam.lights, *yam.keys]:
            yam.delete(element)
        # The stock gripper tops out near 4 N; give it enough force to hold labware.
        for act in yam.actuators:
            if act.name == "gripper":
                act.gainprm[0] = 1000
                act.biasprm[1] = -1000
                act.biasprm[2] = -40
                act.forcerange = [-40, 40]
        link6 = next(b for b in yam.bodies if b.name == "link_6")
        # Looks along the fingers from the side opposite the pads (approximate, not the real D405 mount).
        # Menagerie's grasp_site sits on the wrist axis; the finger pads are offset 44 mm from it.
        link6.add_site(name="pinch_site", pos=[0, -0.044, 0.1297], quat=[1, 0, 0, -1], size=[0.004, 0, 0],
                       rgba=[0, 0.6, 1, 1], group=4)
        link6.add_camera(name="wrist_cam", pos=[0, 0.085, 0.07], xyaxes=[-1, 0, 0, 0, 0.9, 0.44], fovy=75)
        frame = world.add_frame(pos=ARM_BASE[arm])
        spec.attach(yam, prefix=f"{arm}/", frame=frame)

    if hasattr(layout, "add_to"):            # layouts defined in their own modules build their own labware
        layout.add_to(spec)
    elif isinstance(layout, WeighLayout):
        # 68 loose grains: Newton's dense factorisation costs 12 ms a step here, conjugate gradient under 1 ms.
        spec.option.solver = mujoco.mjtSolver.mjSOL_CG
        _add_rack(spec, layout.stock, rgba=[0.15, 0.55, 0.6, 1])
        _add_rack(spec, layout.handover, rgba=[0.6, 0.35, 0.6, 1])
        _add_salt_vial(spec, "salt_vial", layout.vial_start())
        _add_balance(spec, layout.balance)
        eye, target = np.array([0.62, -0.02, 0.5]), np.array([-0.02, -0.02, 0.03])
        forward = (target - eye) / np.linalg.norm(target - eye)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        world.add_camera(name="bench", pos=eye, xyaxes=[*right, *np.cross(right, forward)], fovy=50)
    elif isinstance(layout, SnapLayout):
        _add_rack(spec, layout.rack, rgba=[0.15, 0.55, 0.6, 1])
        for i, (name, cell) in enumerate(layout.tubes.items()):
            _add_snap_tube(spec, name, layout.rack.cell_xy(*cell), layout.yaws[name],
                           TUBE_COLORS[i % len(TUBE_COLORS)])
        # Close view of the rack from the far side of the arm, for videos.
        eye, target = np.array([*layout.rack.xy, 0]) + [0.30, 0.20, 0.26], np.array([*layout.rack.xy, 0.07])
        forward = (target - eye) / np.linalg.norm(target - eye)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        world.add_camera(name="closeup", pos=eye, xyaxes=[*right, *np.cross(right, forward)], fovy=40)
    else:
        _add_rack(spec, layout.source, rgba=[0.15, 0.55, 0.6, 1])
        _add_rack(spec, layout.dest, rgba=[0.6, 0.35, 0.6, 1])
        for i, (name, cell) in enumerate(layout.tubes.items()):
            _add_tube(spec, name, layout.source.cell_xy(*cell), TUBE_COLORS[i % len(TUBE_COLORS)])

    world.add_camera(name="overhead", pos=[0.12, 0, 1.0], xyaxes=[0, -1, 0, 1, 0, 0], fovy=50)
    world.add_camera(name="front", pos=[0.95, 0, 0.45], xyaxes=[0, 1, 0, -0.42, 0, 0.91], fovy=50)
    return spec


def build_model(layout: Layout | SnapLayout | WeighLayout | None = None) -> mujoco.MjModel:
    return build_spec(layout).compile()
