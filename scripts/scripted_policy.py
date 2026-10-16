"""Hand-written baseline: proves the task is solvable through the public env API.

    python scripts/scripted_policy.py --seeds 20
    python scripts/scripted_policy.py --seeds 1 --save previews
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biolab import TubeRackEnv  # noqa: E402

ARM = "right"
GRASP_BELOW_TOP = 0.018   # pinch this far below the tube top
CLEAR_Z = 0.152           # travel height: carried tube clears the standing ones
PREGRASP_OPENING = 0.050
RELEASE_OPENING = 0.050


def policy(env: TubeRackEnv) -> bool:
    obs = env.perceive()
    tube = obs["tubes"][env.task.tube]
    src_yaw = obs["racks"]["rack_source"]["yaw"]
    dst = obs["racks"]["rack_dest"]
    grasp = tube["top"].copy()
    grasp[2] -= GRASP_BELOW_TOP
    place = dst["cells"][env.task.dest_cell].copy()
    place[2] += grasp[2] - tube["bottom"][2] + 0.004  # release 4 mm above the base

    def above(p):
        return [p[0], p[1], CLEAR_Z]

    env.set_gripper(ARM, PREGRASP_OPENING)
    steps = [
        lambda: env.move_to(ARM, above(grasp), src_yaw),
        lambda: env.move_to(ARM, grasp, src_yaw, speed=0.08),
        lambda: env.set_gripper(ARM, 0.0) or True,
        lambda: env.move_to(ARM, above(grasp), src_yaw, speed=0.08),
        lambda: env.move_to(ARM, above(place), dst["yaw"]),
        lambda: env.move_to(ARM, place, dst["yaw"], speed=0.05),
        lambda: env.set_gripper(ARM, RELEASE_OPENING) or True,
        lambda: env.move_to(ARM, above(place), dst["yaw"], speed=0.08),
    ]
    for i, step in enumerate(steps):
        if not step():
            print(f"  step {i} failed (unreachable or not converged)")
            return False
    env.home(ARM)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--save", type=Path, help="folder for before/after images")
    args = ap.parse_args()

    wins = 0
    env = TubeRackEnv(seed=0)
    for seed in range(args.seeds):
        env.reset(seed)
        if args.save:
            _save(env, args.save, f"seed{seed}_before")
        policy(env)
        result = env.evaluate()
        wins += result["success"]
        failed = [k for k, v in result["checks"].items() if not v]
        print(f"seed {seed}: {env.task.instruction} -> {'OK' if result['success'] else 'FAIL ' + str(failed)}")
        if args.save:
            _save(env, args.save, f"seed{seed}_after")
    print(f"success {wins}/{args.seeds}")


def _save(env: TubeRackEnv, folder: Path, stem: str) -> None:
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    for cam in ("overhead", "front", "right/wrist_cam"):
        Image.fromarray(env.render(cam)).save(folder / f"{stem}_{cam.replace('/', '_')}.png")


if __name__ == "__main__":
    main()
