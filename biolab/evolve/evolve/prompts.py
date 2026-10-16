"""Prompt construction and response parsing for the library loop."""
from __future__ import annotations

import inspect
import re

SYSTEM = "You are an expert robot programmer who improves through practice and keeps careful notes."

TASK = """\
You control two I2RT YAM robot arms on a lab bench through the Python API below.
Task: move one sample vial from the source rack into one cell of the destination rack. Call get_task() to learn which vial and which cell; they differ in every episode, and so do the rack positions and angles.
An episode succeeds when that vial stands upright and seated in the target cell, the gripper has let go of it, and the other vial is still in its cell."""

RULES = """\
You are practising. Each round you write code, it is run on several practice layouts, and you are shown what happened. Later, your saved policy is tested once, without feedback, on layouts you have never seen.

You keep a library between rounds:
- skills: a Python module of reusable functions built on the API (grasping, carrying, placing, checks, recovery).
- lessons: short notes about how the robot and the API really behave. Record facts you verified, not guesses.
The library is restored to its best-scoring version if a round scores lower, so only changes that work are kept. It will also be handed to future tasks on this robot, so prefer general skills and lessons over ones tied to this task.

Reply with exactly these three fenced blocks and nothing that matters outside them:

```python skills
# the complete skills module (it replaces the previous one)
```

```markdown lessons
- the complete list of lessons (it replaces the previous one; at most 25 bullets)
```

```python policy
# the script run in each episode; API functions and your skills are already in scope
```

The API functions are in scope without imports; import numpy yourself if you use it. print() output is returned to you as feedback, so print what you need to diagnose failures (return values of move_to, poses before and after a grasp). Do not use code fences other than the three above."""


def api_doc(functions: dict) -> str:
    parts = []
    for name, fn in functions.items():
        doc = inspect.getdoc(fn) or ""
        parts.append(f"{name}{inspect.signature(fn)}\n" + "\n".join(f"    {line}" for line in doc.splitlines()))
    return "\n\n".join(parts)


def format_results(results: list[dict]) -> str:
    lines = []
    for r in results:
        failed = [name for name, ok in r["checks"].items() if not ok]
        lines.append(f"Layout {r['seed']}: {'SUCCESS' if r['success'] else 'FAILED ' + ', '.join(failed)}"
                     f"  ({r['task']}; {r['sim_seconds']} s)")
        if r["error"]:
            lines.append("  error:\n" + "\n".join(f"    {line}" for line in r["error"].splitlines()))
        if r["stdout"].strip():
            lines.append("  stdout:\n" + "\n".join(f"    {line}" for line in r["stdout"].rstrip().splitlines()))
    return "\n".join(lines)


def build_messages(functions: dict, library: dict, last: dict | None, history: list[dict]) -> list[dict]:
    """`library` is the best version so far; `last` is the previous round (or None)."""
    text = [TASK, "", "API:", "", api_doc(functions), "", RULES, "", "## Current library", ""]
    if library["skills"].strip() or library["lessons"].strip():
        text += ["```python skills", library["skills"].rstrip(), "```", "",
                 "```markdown lessons", library["lessons"].rstrip(), "```"]
    else:
        text.append("(empty: this is the first round)")
    if history:
        text += ["", "## Practice so far", ""]
        text += [f"Round {h['iteration']}: {h['successes']}/{h['episodes']} layouts"
                 f"{'' if h['kept'] else '  (scored lower than the best round; library changes discarded)'}"
                 for h in history]
    if last is not None:
        text += ["", f"## Your previous round (round {last['iteration']})", ""]
        if last.get("parse_error"):
            text.append(f"Your reply could not be used: {last['parse_error']}")
        else:
            text += ["```python policy", last["policy"].rstrip(), "```", "", "Results:", format_results(last["results"])]
            if not last["kept"]:
                text.append("\nThis round scored lower than the best one, so the library above is the earlier, "
                            "better version, not the one you wrote last round.")
    text += ["", "Write the next round."]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(text)}]


_BLOCK = r"```[a-zA-Z]*[ \t]+{tag}[ \t]*\r?\n(.*?)```"


def parse_response(response: str) -> dict:
    """Extract the three blocks. Missing blocks come back as None."""
    out = {}
    for tag in ("skills", "lessons", "policy"):
        matches = re.findall(_BLOCK.format(tag=tag), response, re.S)
        out[tag] = matches[-1].strip() + "\n" if matches else None
    return out
