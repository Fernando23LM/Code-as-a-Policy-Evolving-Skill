"""Run one episode of model-written code and report what happened."""
from __future__ import annotations

import contextlib
import io
import signal
import traceback

from ..api import TubeRackApi
from ..env import CONTROL_DT, TubeRackEnv

SIM_BUDGET_S = 180.0    # simulated seconds per episode
WALL_BUDGET_S = 300     # real seconds per episode (only enforced where SIGALRM exists)
STDOUT_LIMIT = 1500     # characters of stdout kept for feedback


class BudgetExceeded(RuntimeError):
    pass


def _compact_error(exc: BaseException) -> str:
    """Exception plus the frames inside the model's own code, with source lines."""
    lines = []
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename in ("skills.py", "policy.py"):
            lines.append(f"{frame.filename}:{frame.lineno} in {frame.name}")
    lines.append(f"{type(exc).__name__}: {exc}")
    return "\n".join(lines)


def run_episode(env: TubeRackEnv, seed: int, skills: str, policy: str) -> dict:
    """Reset to `seed`, run `skills` then `policy` with the API in scope, and judge the result.

    The code sees only the API functions, not the environment or the judge. This is
    name hiding, not a sandbox: only run code from models you are willing to run.
    """
    env.on_tick = None
    env.reset(seed)
    ticks = 0

    def on_tick() -> None:
        nonlocal ticks
        ticks += 1
        if ticks * CONTROL_DT > SIM_BUDGET_S:
            raise BudgetExceeded(f"simulation budget of {SIM_BUDGET_S:.0f} s exceeded")

    def on_alarm(signum, frame):
        raise BudgetExceeded(f"wall-clock budget of {WALL_BUDGET_S} s exceeded")

    env.on_tick = on_tick
    namespace = {"__name__": "__policy__", **TubeRackApi(env).functions()}
    stdout, error = io.StringIO(), None
    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        previous = signal.signal(signal.SIGALRM, on_alarm)
        signal.alarm(WALL_BUDGET_S)
    try:
        with contextlib.redirect_stdout(stdout):
            exec(compile(skills, "skills.py", "exec"), namespace)
            exec(compile(policy, "policy.py", "exec"), namespace)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # model code may raise anything, including SystemExit
        error = _compact_error(exc)
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        env.on_tick = None

    env.step(1.0)  # let things come to rest before judging
    verdict = env.evaluate()
    return {
        "seed": seed,
        "task": env.task.instruction,
        "success": bool(verdict["success"]),
        "checks": verdict["checks"],
        "error": error,
        "stdout": stdout.getvalue()[-STDOUT_LIMIT:],
        "sim_seconds": round(ticks * CONTROL_DT, 1),
    }
