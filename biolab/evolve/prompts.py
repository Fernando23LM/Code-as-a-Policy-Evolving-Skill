"""Prompt construction and response parsing for the library loop."""
from __future__ import annotations

import inspect
import re

SYSTEM = "You are an expert robot programmer who improves through practice and keeps careful notes."

RULES = """\
You are practising. Each round you write code, it is run on several practice layouts, and you are shown what happened. Later, your saved policy is tested once, without feedback, on layouts you have never seen.

You keep a library between rounds:
- skills: a Python module of reusable functions built on the API (grasping, carrying, placing, checks, recovery).
- lessons: short notes about how the robot and the API really behave. Each lesson names a cause and what to do about it, and is written only after you tested it; do not repeat the API text, and do not log scores or which layouts failed in which round.
When the same layouts keep failing, work out what they have in common that the successful ones do not, and test one change at a time.
The library is restored to its best-scoring version if a round scores lower, so only changes that work are kept. Rounds are ranked by layouts solved, then by partial credit (0 to 1 per layout: how far a failed attempt got). It will also be handed to future tasks on this robot, so prefer general skills and lessons over ones tied to this task.

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

The API functions are in scope without imports; import numpy yourself if you use it. print() output is returned to you as feedback, so print what you need to diagnose failures (the arguments you passed to move_to and what it returned, poses before and after a grasp). Only the first 1500 and last 2500 characters of each layout's output are kept, so keep prints short. Do not use code fences other than the three above."""


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
        partial = "" if r["success"] or "progress" not in r else f", partial credit {r['progress']:.2f}"
        note = f"; {r['note']}" if r.get("note") else ""
        lines.append(f"Layout {r['seed']}: {'SUCCESS' if r['success'] else 'FAILED ' + ', '.join(failed)}{partial}"
                     f"  ({r['task']}; {r['sim_seconds']} s{note})")
        if r["error"]:
            lines.append("  error:\n" + "\n".join(f"    {line}" for line in r["error"].splitlines()))
        if r["stdout"].strip():
            lines.append("  stdout:\n" + "\n".join(f"    {line}" for line in r["stdout"].rstrip().splitlines()))
    return "\n".join(lines)


def modules_doc(modules: dict) -> list[str]:
    """Prompt section for fixed modules: their code, and the script that used each as an example."""
    text = ["## Fixed modules", "",
            "These skill modules were learned on earlier tasks on this robot. They are already loaded and you "
            "cannot change them. Call their functions through the module name, for example "
            f"{next(iter(modules))}.some_function(...). They were written against the same API, but for their own "
            "task: read them before relying on them, and write your own code beside them in your skills block "
            "where they do not fit. Each module is followed by the script that used it on its own task."]
    for name, lib in modules.items():
        text += ["", f"### module {name}", "", "```python", lib["skills"].rstrip(), "```", "",
                 f"Script that used {name} (for reference; names it defines are not available to you):", "",
                 "```python", lib["policy"].rstrip(), "```"]
    return text + [""]


def build_messages(task: str, functions: dict, library: dict, last: dict | None, history: list[dict],
                   modules: dict | None = None) -> list[dict]:
    """`task` is the task's prompt text; `library` the best version so far; `last` the previous round (or None).

    `modules` are fixed skill libraries from earlier tasks (name -> {"skills", "lessons", "policy"}).
    """
    text = [task, "", "API:", "", api_doc(functions), "", RULES, ""]
    if modules:
        text += modules_doc(modules)
    text += ["## Current library", ""]
    if library["skills"].strip() or library["lessons"].strip():
        text += ["```python skills", library["skills"].rstrip(), "```", "",
                 "```markdown lessons", library["lessons"].rstrip(), "```"]
    else:
        text.append("(empty: this is the first round)")
    if history:
        text += ["", "## Practice so far", ""]
        text += [f"Round {h['iteration']}: {h['successes']}/{h['episodes']} layouts"
                 f"{'' if h.get('progress') is None else ', partial credit ' + format(h['progress'], '.2f')}"
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


_FENCE_LINE = re.compile(r"^```[ \t]*([a-zA-Z]*)[ \t]*([a-zA-Z]*)[ \t]*$")


def _blocks(response: str) -> list[tuple[str, str, str]]:
    """Fenced blocks as (language, label, body), read line by line.

    A fence line that names a language while a block is open starts the next block: models sometimes
    forget to close one. A block still open at the end runs to the end.
    """
    blocks, current, body = [], None, []
    for line in response.splitlines():
        fence = _FENCE_LINE.match(line.strip())
        if fence is None:
            if current is not None:
                body.append(line)
            continue
        lang, tag = fence.group(1), fence.group(2)
        if current is not None:
            blocks.append((*current, "\n".join(body)))
            current, body = None, []
            if not lang:                      # a plain closing fence
                continue
        current = (lang, tag)
    if current is not None:
        blocks.append((*current, "\n".join(body)))
    return blocks


def parse_response(response: str) -> dict:
    """Extract the three blocks. Missing blocks come back as None.

    Models sometimes drop the label (```python instead of ```python policy), forget a closing fence or
    mangle the last one. Labelled blocks win; unlabelled ones are then matched by language and order
    (python: skills then policy, or policy alone; markdown or plain: lessons).
    """
    out = {"skills": None, "lessons": None, "policy": None}
    loose = {"python": [], "text": []}
    for lang, tag, body in _blocks(response):
        body = re.sub(r"\n`{1,2}[^`\n]*\Z", "", body).strip() + "\n"      # drop a mangled closing fence
        lang, tag = lang.lower(), tag.lower()
        if tag in out:
            out[tag] = body
        elif lang in out and not tag:                                      # ```policy
            out[lang] = body
        else:
            loose["python" if lang in ("python", "py") else "text"].append(body)
    code = loose["python"]
    if out["policy"] is None and code:
        out["policy"] = code.pop()                                         # the script comes last
    if out["skills"] is None and code:
        out["skills"] = code[0]
    if out["lessons"] is None and loose["text"]:
        out["lessons"] = loose["text"][-1]
    return out
