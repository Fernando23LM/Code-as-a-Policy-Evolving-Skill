"""Practice videos: one mp4 per round, every practice layout back to back at 4x speed."""
from __future__ import annotations

from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 640, 480
FRAME_EVERY = 10        # control ticks between frames: 5 frames per simulated second
FPS = 20
HOLD_FRAMES = 20        # the verdict stays on screen for a second


class Recorder:
    """Needs an OpenGL context (MUJOCO_GL=egl on a headless node), imageio-ffmpeg and Pillow."""

    def __init__(self, path: Path, env, camera: str, title: str) -> None:
        import imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        self.env, self.camera, self.title, self.label = env, camera, title, ""
        self._writer = imageio.get_writer(path, fps=FPS, macro_block_size=16)
        self._clock = None

    def start(self, seed: int) -> None:
        self.label = f"{self.title}   layout {seed}"
        self._clock = getattr(self.env, "clock", None)
        self.frame()

    def frame(self, verdict: str = "", color=(255, 255, 255)) -> np.ndarray:
        from PIL import Image, ImageDraw, ImageFont

        image = Image.fromarray(self.env.render(self.camera, WIDTH, HEIGHT))
        draw, font = ImageDraw.Draw(image), ImageFont.load_default(size=20)
        label = self.label
        target = getattr(self.env.task, "target_g", None)
        if target is not None:                       # weighing tasks: show the display and the target
            label += f"   {self.env.balance_reading():.2f} g of {target:.1f} g"
        lines = [label]
        status = getattr(self.env, "video_status", None)
        if status is not None:                       # pipetting tasks: clock, cells, pipette contents
            lines += status()
        for k, line in enumerate(lines):
            draw.text((10, 8 + 26 * k), line, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        if verdict:
            draw.text((10, 10 + 26 * len(lines)), verdict, fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        frame = np.asarray(image)
        self._writer.append_data(frame)
        clock = getattr(self.env, "clock", None)
        if clock is not None and self._clock is not None and clock > self._clock:
            for _ in range(HOLD_FRAMES - 1):         # incubation jumps the clock: hold so the jump can be read
                self._writer.append_data(frame)
        self._clock = clock
        return frame

    def end(self, result: dict) -> None:
        failed = [name for name, ok in result["checks"].items() if not ok]
        verdict = "SUCCESS" if result["success"] else "FAILED: " + ", ".join(failed)
        frame = self.frame(verdict, (90, 230, 110) if result["success"] else (255, 110, 90))
        for _ in range(HOLD_FRAMES - 1):
            self._writer.append_data(frame)

    def close(self) -> None:
        self._writer.close()


def join_videos(parts: list[Path], out: Path) -> None:
    """Concatenate per-layout videos into one file and remove the parts."""
    import imageio

    with imageio.get_writer(out, fps=FPS, macro_block_size=16) as writer:
        for part in parts:
            with imageio.get_reader(part) as reader:
                for frame in reader:
                    writer.append_data(frame)
    for part in parts:
        part.unlink()

