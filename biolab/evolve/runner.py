"""Run one episode of model-written code and report what happened."""
from __future__ import annotations

import contextlib
import io
import signal
import traceback
import types

from ..api import TubeRackApi
from ..env import CONTROL_DT, TubeRackEnv
from .video import FRAME_EVERY

SIM_BUDGET_S = 180.0    # simulated seconds per episode
WALL_BUDGET_S = 300     # real seconds per episode (only enforced where SIGALRM exists)
STDOUT_HEAD = 1500      # characters of stdout kept for feedback: the start, where a policy says what it
STDOUT_TAIL = 2500      # set out to do, and the end, where it says how things finished


class BudgetExceeded(RuntimeError):
    pass


def _compact_error(exc: BaseException, files: tuple[str, ...] = ("skills.py", "policy.py")) -> str:
    """Exception plus the frames inside the model's own code (and its fixed modules), with source lines."""
    lines = []
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename in files:
            lines.append(f"{frame.filename}:{frame.lineno} in {frame.name}")
    lines.append(f"{type(exc).__name__}: {exc}")
    return "\n".join(lines)


def _clip(text: str) -> str:
    if len(text) <= STDOUT_HEAD + STDOUT_TAIL:
        return text
    omitted = len(text) - STDOUT_HEAD - STDOUT_TAIL
    return f"{text[:STDOUT_HEAD]}\n... [{omitted} characters omitted] ...\n{text[-STDOUT_TAIL:]}"


def run_episode(env: TubeRackEnv, seed: int, skills: str, policy: str, api: type = TubeRackApi,
                recorder=None, modules: dict[str, str] | None = None) -> dict:
    """Reset to `seed`, run `skills` then `policy` with the API in scope, and judge the result.

    The code sees only the API functions, not the environment or the judge. This is
    name hiding, not a sandbox: only run code from models you are willing to run.
    `recorder` (biolab.evolve.video.Recorder) films the episode if given. `modules` maps a name to the
    source of a skill library learned earlier: each is loaded on its own, so their helpers cannot clash,
    and reaches the code as an object, e.g. `handover.pick(...)`.
    """
    env.on_tick = None
    env.reset(seed)
    ticks = 0
    sim_budget = getattr(env, "SIM_BUDGET_S", SIM_BUDGET_S)      # an environment may ask for more
    wall_budget = getattr(env, "WALL_BUDGET_S", WALL_BUDGET_S)
    if recorder is not None:
        wall_budget *= 2           # rendering is slow when many layouts share one GPU; it must not cut a policy short
    every = getattr(env, "VIDEO_EVERY", FRAME_EVERY)   # long tasks are filmed more sparsely
    if recorder is not None:
        recorder.start(seed)

    def on_tick() -> None:
        nonlocal ticks
        ticks += 1
        if recorder is not None and ticks % every == 0:
            recorder.frame()
        if ticks * CONTROL_DT > sim_budget:
            raise BudgetExceeded(f"simulation budget of {sim_budget:.0f} s exceeded")

    def on_alarm(signum, frame):
        raise BudgetExceeded(f"wall-clock budget of {wall_budget} s exceeded")

    env.on_tick = on_tick
    functions = api(env).functions()
    namespace = {"__name__": "__policy__", **functions}
    modules = modules or {}
    files = ("skills.py", "policy.py", *(f"{name}.py" for name in modules))
    stdout, error = io.StringIO(), None
    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        previous = signal.signal(signal.SIGALRM, on_alarm)
        signal.alarm(wall_budget)
    try:
        with contextlib.redirect_stdout(stdout):
            for name, source in modules.items():
                scope = {"__name__": name, **functions}
                exec(compile(source, f"{name}.py", "exec"), scope)
                namespace[name] = types.SimpleNamespace(**{k: v for k, v in scope.items() if not k.startswith("__")})
            exec(compile(skills, "skills.py", "exec"), namespace)
            exec(compile(policy, "policy.py", "exec"), namespace)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # model code may raise anything, including SystemExit
        error = _compact_error(exc, files)
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        env.on_tick = None

    env.step(1.0)  # let things come to rest before judging
    verdict = env.evaluate()
    if recorder is not None:
        recorder.end(verdict)
    return {
        "seed": seed,
        "task": env.task.instruction,
        "success": bool(verdict["success"]),
        "checks": {name: bool(ok) for name, ok in verdict["checks"].items()},   # numpy bools do not go to JSON
        "progress": float(verdict.get("progress", verdict["success"])),   # partial credit where the judge gives it
        "note": verdict.get("note", ""),
        "error": error,
        "stdout": _clip(stdout.getvalue()),
        "sim_seconds": round(ticks * CONTROL_DT, 1),
    }
