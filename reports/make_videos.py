"""Videos (MP4 and GIF) of the reference policy on the protein assay and the cell-culture task.

    MUJOCO_GL=egl python reports/make_videos.py
"""
import sys
from pathlib import Path

import imageio
import mujoco
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import biolab.evolve.runner as runner  # noqa: E402
from biolab.tasks import TASKS  # noqa: E402

OUT = Path(__file__).resolve().parent / "videos"
WIDTH, HEIGHT = 640, 480


def protein_status(env):
    p = env.pipette
    text = f"pipette: {p.volume:.0f} uL"
    if p.volume > 0.5:
        text += f" at {p.conc:.3f} mg/mL"
    if env.reported is not None:
        text += f"   reported {env.reported:.3f} mg/mL (true {env.task.sample:.3f})"
    return text


def culture_status(env):
    cells = "  ".join(f"{d}: {env.attached[d] / 1e6:.2f}M" for d in ("A", "B", "C"))
    return f"day {env.clock / 24:.1f} ({env.clock:.0f} h)   attached cells  {cells}"


# task: (title, seed, ticks per frame, frames per second, status line, camera look-at, distance, azimuth, elevation)
SHOTS = {
    "protein_assay": ("Protein assay: standards, sample, plate read", 0, 12, 20, protein_status,
                      (0.04, -0.30, 0.02), 0.40, 160, -42),
    "cell_culture": ("Cell culture: one simulated week", 0, 25, 25, culture_status,
                     (0.05, -0.29, 0.02), 0.44, 160, -42),
}


class Recorder:
    def __init__(self, env, title, status, lookat, distance, azimuth, elevation):
        self.env, self.title, self.status, self.frames = env, title, status, []
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:], self.cam.distance, self.cam.azimuth, self.cam.elevation = lookat, distance, azimuth, elevation
        self.renderer, self.font = None, ImageFont.load_default(size=17)

    def start(self, seed):
        self.renderer = mujoco.Renderer(self.env.model, HEIGHT, WIDTH)
        self.frame()

    def frame(self, verdict="", color=(255, 255, 255)):
        self.renderer.update_scene(self.env.data, camera=self.cam)
        image = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(image)
        for i, (text, fill) in enumerate(((self.title, (255, 255, 255)), (self.status(self.env), (255, 235, 160)),
                                          (verdict, color))):
            if text:
                draw.text((10, 8 + 24 * i), text, fill=fill, font=self.font, stroke_width=2, stroke_fill=(0, 0, 0))
        self.frames.append(image)

    def end(self, result):
        ok = result["success"]
        for _ in range(40):
            self.frame("SUCCESS" if ok else "FAILED", (90, 230, 110) if ok else (255, 110, 90))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for task, (title, seed, every, fps, status, *camera) in SHOTS.items():
        runner.FRAME_EVERY = every
        spec = TASKS[task]
        env = spec.env(seed=seed)
        rec = Recorder(env, title, status, *camera)
        result = runner.run_episode(env, seed, "", spec.reference, spec.api, rec)
        rec.close()
        mp4 = OUT / f"{task}.mp4"
        with imageio.get_writer(mp4, fps=fps, macro_block_size=16) as writer:
            for f in rec.frames:
                writer.append_data(__import__("numpy").asarray(f))
        small = [f.resize((480, 360)) for f in rec.frames[::2]]
        palette = Image.new("RGB", (480, 720))
        palette.paste(small[len(small) // 2], (0, 0)); palette.paste(small[-1], (0, 360))
        palette = palette.quantize(colors=160, method=Image.Quantize.MEDIANCUT)
        gif = OUT / f"{task}.gif"
        frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in small]
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=int(2000 / fps), loop=0, optimize=True)
        print(f"{task}: {'success' if result['success'] else 'FAILED'} | {result['note'][:160]} | "
              f"{len(rec.frames)} frames, {len(rec.frames) / fps:.0f} s, mp4 {mp4.stat().st_size / 1e6:.1f} MB, "
              f"gif {gif.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
