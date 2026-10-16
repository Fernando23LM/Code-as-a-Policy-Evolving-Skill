"""Film the hand-written reference policy of a task on a few layouts, one video per task.

    MUJOCO_GL=egl python scripts/record_reference.py protein_assay cell_culture --seeds 0-2

Writes previews/<task>_reference.mp4 plus a few stills from it, and prints each layout's verdict.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biolab.evolve.loop import Config, run_layouts  # noqa: E402
from biolab.tasks import TASKS  # noqa: E402


def seed_list(text: str) -> list[int]:
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="+", choices=sorted(TASKS))
    parser.add_argument("--seeds", type=seed_list, default=seed_list("0-2"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "previews")
    args = parser.parse_args()

    import imageio

    for task in args.tasks:
        start = time.time()
        video = args.out / f"{task}_reference.mp4"
        cfg = Config(run_dir=str(args.out), task=task, workers=args.workers)
        results = run_layouts(cfg, None, args.seeds, "", TASKS[task].reference, video, f"{task} reference")
        for r in results:
            print(task, r["seed"], "OK  " if r["success"] else "FAIL", r["note"][:160], (r["error"] or "")[-200:],
                  flush=True)
        with imageio.get_reader(video) as reader:
            n = reader.count_frames()
            for share in (0.1, 0.5, 0.95):
                imageio.imwrite(args.out / f"{task}_reference_{int(share * 100):02d}.png", reader.get_data(int(n * share)))
        print(f"{task}: {sum(r['success'] for r in results)}/{len(results)}, {n} frames ({n / 20:.0f} s of video), "
              f"{video.stat().st_size // 1024} kB, {time.time() - start:.0f} s wall", flush=True)


if __name__ == "__main__":
    main()
