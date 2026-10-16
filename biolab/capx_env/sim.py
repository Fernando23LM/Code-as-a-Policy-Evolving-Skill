"""CaP-X low-level environment around TubeRackEnv."""
from __future__ import annotations

from typing import Any

import numpy as np
from capx.envs.base import BaseEnv

from ..env import CONTROL_DT, TubeRackEnv

# CaP-X's vision APIs look for the main camera under this key.
MAIN_CAMERA_KEY = "robot0_robotview"
CAMERAS = {MAIN_CAMERA_KEY: "overhead", "front": "front", "right_wrist": "right/wrist_cam",
           "left_wrist": "left/wrist_cam"}
VIDEO_CAMERA = "front"
WRIST_CAMERA = "right/wrist_cam"
FRAME_EVERY = 5  # control ticks per video frame (10 fps of simulated time)


class TubeRackLowLevel(BaseEnv):
    """Two YAM arms, a source rack with vials and a destination rack.

    CaP-X resets with seed = trial index; `seed_offset` shifts that so practice
    and test runs can use disjoint layouts.
    """

    def __init__(self, privileged: bool = False, enable_render: bool = True, viser_debug: bool = False,
                 max_steps: int = 9000, seed_offset: int = 0) -> None:
        self.privileged = privileged
        self.enable_render = enable_render
        self.max_steps = max_steps  # control ticks; 9000 = 3 minutes of simulated time
        self.seed_offset = seed_offset
        self.sim = TubeRackEnv(seed=seed_offset)
        self._sim_step_count = 0
        self._record_frames = False
        self._record_wrist_camera = False
        self._frame_buffer: list[np.ndarray] = []
        self._wrist_frame_buffer: list[np.ndarray] = []
        self.sim.on_tick = self._on_tick

    # ------------------------------------------------------------- gym interface
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.sim.on_tick = None  # settling during reset is not part of the episode
        self.sim.reset(None if seed is None else seed + self.seed_offset)
        self.sim.on_tick = self._on_tick
        self._sim_step_count = 0
        self._frame_buffer.clear()
        self._wrist_frame_buffer.clear()
        return self.get_observation(), {"task": self.sim.task.instruction}

    def step(self, action: Any):
        """Let the simulation run for one control tick. Policies act through the API instead."""
        self.sim.step(CONTROL_DT)
        reward = self.compute_reward()
        return self.get_observation(), reward, reward == 1.0, self._sim_step_count >= self.max_steps, {}

    def get_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = {"task": self.sim.task.instruction}
        if self.enable_render:
            for key, camera in CAMERAS.items():
                obs[key] = self.sim.camera_observation(camera)
        if self.privileged:
            obs["state"] = self.sim.perceive()
        return obs

    def compute_reward(self) -> float:
        return 1.0 if self.task_completed() else 0.0

    def task_completed(self) -> bool:
        return bool(self.sim.evaluate()["success"])

    # -------------------------------------------------------------------- video
    def _on_tick(self) -> None:
        self._sim_step_count += 1
        if self._sim_step_count > self.max_steps:
            raise RuntimeError(f"Simulation budget of {self.max_steps * CONTROL_DT:.0f} s exceeded")
        if self._record_frames and self._sim_step_count % FRAME_EVERY == 0:
            self._record_frame()

    def _record_frame(self) -> None:
        self._frame_buffer.append(self.sim.render(VIDEO_CAMERA).copy())
        if self._record_wrist_camera:
            self._wrist_frame_buffer.append(self.sim.render(WRIST_CAMERA).copy())

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True, wrist_camera: bool = False) -> None:
        self._record_frames = enabled
        self._record_wrist_camera = enabled and wrist_camera
        if clear:
            self._frame_buffer.clear()
            self._wrist_frame_buffer.clear()
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = list(self._frame_buffer)
        if clear:
            self._frame_buffer = []
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return self._frame_buffer[start:end]

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = list(self._wrist_frame_buffer)
        if clear:
            self._wrist_frame_buffer = []
        return frames

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return self._wrist_frame_buffer[start:end]

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        return self.sim.render(VIDEO_CAMERA)

    def render_wrist(self) -> np.ndarray:
        return self.sim.render(WRIST_CAMERA)
