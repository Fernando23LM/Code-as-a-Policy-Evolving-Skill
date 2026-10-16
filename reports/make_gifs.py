"""Animated GIFs of the reference policy on each task in the session summary.

    MUJOCO_GL=egl python reports/make_gifs.py [--check]

One layout per task, filmed from a camera aimed at where the work happens, at 6x speed.
--check only saves the first frame of each, to aim the cameras.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biolab.evolve.runner import run_episode  # noqa: E402
from biolab.tasks import TASKS  # noqa: E402

OUT = Path(__file__).resolve().parent / "gifs"
WIDTH, HEIGHT = 480, 360
EVERY = 15            # control ticks between frames: one frame per 0.3 simulated seconds
FRAME_MS = 50         # 6x speed
# task: (title, seed, camera look-at point, distance, azimuth, elevation)
SHOTS = {
    "tube_rack": ("Task 1: move a vial between racks", 0, (0.02, -0.28, 0.05), 0.50, 165, -38),
    "snap_cap": ("Task 2: open a snap-cap tube", 0, (0.03, -0.29, 0.07), 0.36, 150, -28),
    "weigh_handover": ("Task 3a: hand the salt vial to the middle rack", 0, (-0.04, 0.17, 0.06), 0.62, 180, -38),
    "weigh_pour": ("Task 3b: pour salt to a target mass", 0, (0.01, -0.15, 0.05), 0.58, 172, -32),
    "weigh_salt": ("Task 3: weigh out salt with both arms", 0, (-0.03, -0.02, 0.05), 0.95, 180, -42),
}


class GifRecorder:
    """Same interface as biolab.evolve.video.Recorder, but frames go to a list for a GIF."""

    def __init__(self, env, title, lookat, distance, azimuth, elevation):
        self.env, self.title, self.frames, self.label = env, title, [], ""
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:], self.cam.distance, self.cam.azimuth, self.cam.elevation = lookat, distance, azimuth, elevation
        self.renderer = None
        self.font = ImageFont.load_default(size=15)

    def start(self, seed):
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = mujoco.Renderer(self.env.model, HEIGHT, WIDTH)
        self.frame()

    def frame(self, verdict="", color=(255, 255, 255)):
        self.renderer.update_scene(self.env.data, camera=self.cam)
        image = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(image)
        text = self.title
        target = getattr(self.env.task, "target_g", None)
        if target is not None:
            text += f"   balance {self.env.balance_reading():.2f} g of {target:.1f} g"
        draw.text((8, 6), text, fill=(255, 255, 255), font=self.font, stroke_width=2, stroke_fill=(0, 0, 0))
        if verdict:
            draw.text((8, 28), verdict, fill=color, font=self.font, stroke_width=2, stroke_fill=(0, 0, 0))
        self.frames.append(image)

    def end(self, result):
        ok = result["success"]
        for _ in range(30):                          # hold the verdict for 1.5 s
            self.frame("SUCCESS" if ok else "FAILED", (90, 230, 110) if ok else (255, 110, 90))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def main():
    import biolab.evolve.runner as runner
    runner.FRAME_EVERY = EVERY
    check = "--check" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    for task, (title, seed, *camera) in SHOTS.items():
        spec = TASKS[task]
        env = spec.env(seed=seed)
        rec = GifRecorder(env, title, *camera)
        if check:
            rec.start(seed); rec.frames[0].save(OUT / f"{task}_check.png"); rec.close(); print(task, "first frame saved"); continue
        result = run_episode(env, seed, "", spec.reference, spec.api, rec)
        rec.close()
        palette = rec.frames[len(rec.frames) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in rec.frames]
        path = OUT / f"{task}.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=FRAME_MS, loop=0, optimize=True)
        print(f"{task}: {'success' if result['success'] else 'FAILED'}, {len(frames)} frames, "
              f"{len(frames) * FRAME_MS / 1000:.0f} s, {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
