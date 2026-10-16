"""Practice / keep-what-works / test loop.

Run directory layout (everything is plain files, so a job can stop and resume):

    run_dir/
      config.json
      round_000/  prompt.txt response.txt skills.py lessons.md policy.py results.json
      round_001/  ...
      library/    skills.py lessons.md policy.py   <- best round so far
      test.json   summary.txt
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..api import TubeRackApi
from ..env import TubeRackEnv
from .llm import LLM
from .prompts import build_messages, parse_response
from .runner import run_episode

EMPTY_LIBRARY = {"skills": "", "lessons": ""}


@dataclass
class Config:
    run_dir: str
    rounds: int = 8
    practice_seeds: list[int] = field(default_factory=lambda: [100, 101, 102, 103])
    test_seeds: list[int] = field(default_factory=lambda: list(range(1000, 1020)))
    inherit: str | None = None       # library folder from an earlier run to start from
    memory: bool = True              # False = independent restarts: no library, no feedback
    stop_when_perfect: bool = True
    model: str = ""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def load_library(folder: Path | None) -> dict:
    if folder is None or not folder.exists():
        return dict(EMPTY_LIBRARY)
    return {"skills": _read(folder / "skills.py"), "lessons": _read(folder / "lessons.md")}


def _load_rounds(run_dir: Path) -> list[dict]:
    rounds = []
    for folder in sorted(run_dir.glob("round_*")):
        if (folder / "results.json").exists():
            rounds.append(json.loads(_read(folder / "results.json")))
    return rounds


def _best(rounds: list[dict]) -> dict | None:
    kept = [r for r in rounds if r["kept"]]
    return kept[-1] if kept else None


def practice(cfg: Config, llm: LLM, env: TubeRackEnv | None = None, log=print) -> list[dict]:
    """Run practice rounds until the budget is spent. Resumes from finished rounds on disk."""
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write(run_dir / "config.json", json.dumps(asdict(cfg), indent=2))
    env = env or TubeRackEnv(seed=0)
    functions = TubeRackApi(env).functions()
    inherited = load_library(Path(cfg.inherit)) if cfg.inherit else dict(EMPTY_LIBRARY)
    rounds = _load_rounds(run_dir)

    while len(rounds) < cfg.rounds:
        best = _best(rounds)
        if cfg.stop_when_perfect and best and best["successes"] == best["episodes"]:
            log(f"round {best['iteration']} solved every practice layout; stopping practice early")
            break
        i = len(rounds)
        folder = run_dir / f"round_{i:03d}"
        library = load_library(run_dir / "library") if best else inherited
        if cfg.memory:
            messages = build_messages(functions, library, rounds[-1] if rounds else None,
                                      [{k: r[k] for k in ("iteration", "successes", "episodes", "kept")} for r in rounds])
        else:
            messages = build_messages(functions, dict(EMPTY_LIBRARY), None, [])
        _write(folder / "prompt.txt", messages[-1]["content"])
        response = llm(messages)
        _write(folder / "response.txt", response)

        parsed = parse_response(response)
        record = {"iteration": i, "episodes": len(cfg.practice_seeds), "successes": 0, "kept": False,
                  "policy": parsed["policy"] or "", "results": [], "parse_error": None}
        skills = parsed["skills"] if parsed["skills"] is not None else library["skills"]
        lessons = parsed["lessons"] if parsed["lessons"] is not None else library["lessons"]
        if parsed["policy"] is None:
            record["parse_error"] = "no ```python policy block found"
        else:
            for name, source in (("skills", skills), ("policy", parsed["policy"])):
                try:
                    compile(source, f"{name}.py", "exec")
                except SyntaxError as exc:
                    record["parse_error"] = f"{name} block has a syntax error on line {exc.lineno}: {exc.msg}"
        if record["parse_error"] is None:
            record["results"] = [run_episode(env, seed, skills, parsed["policy"]) for seed in cfg.practice_seeds]
            record["successes"] = sum(r["success"] for r in record["results"])
            # Keep what works: a round replaces the library only if it scores at least as well.
            record["kept"] = record["successes"] >= (best["successes"] if best else 0)
        _write(folder / "skills.py", skills)
        _write(folder / "lessons.md", lessons)
        _write(folder / "policy.py", record["policy"])
        if record["kept"]:
            for name in ("skills.py", "lessons.md", "policy.py"):
                _write(run_dir / "library" / name, _read(folder / name))
        _write(folder / "results.json", json.dumps(record, indent=2))  # written last: marks the round complete
        rounds.append(record)
        log(f"round {i}: {record['successes']}/{record['episodes']} practice layouts"
            f"{'' if record['kept'] else '  (not kept)'}"
            f"{'  [' + record['parse_error'] + ']' if record['parse_error'] else ''}")
    return rounds


def test(cfg: Config, env: TubeRackEnv | None = None, log=print) -> dict:
    """Score the frozen library policy on held-out layouts. No model calls, no feedback."""
    run_dir = Path(cfg.run_dir)
    library = run_dir / "library"
    if not (library / "policy.py").exists():
        result = {"successes": 0, "episodes": len(cfg.test_seeds), "results": [], "note": "no round was kept"}
    else:
        env = env or TubeRackEnv(seed=0)
        skills, policy = _read(library / "skills.py"), _read(library / "policy.py")
        results = [run_episode(env, seed, skills, policy) for seed in cfg.test_seeds]
        result = {"successes": sum(r["success"] for r in results), "episodes": len(results), "results": results}
    _write(run_dir / "test.json", json.dumps(result, indent=2))
    log(f"TEST: {result['successes']}/{result['episodes']} held-out layouts")
    return result


def summarize(cfg: Config, rounds: list[dict], result: dict) -> str:
    lines = [f"run: {cfg.run_dir}", f"model: {cfg.model}",
             f"mode: {'library loop' if cfg.memory else 'independent restarts'}"
             f"{', inherited ' + cfg.inherit if cfg.inherit else ''}", ""]
    lines += [f"round {r['iteration']}: {r['successes']}/{r['episodes']}{'  kept' if r['kept'] else ''}" for r in rounds]
    lines += ["", f"model calls: {len(rounds)}", f"test: {result['successes']}/{result['episodes']} held-out layouts"]
    failed: dict[str, int] = {}
    for r in result["results"]:
        for name, ok in r["checks"].items():
            if not ok:
                failed[name] = failed.get(name, 0) + 1
    if failed:
        lines.append("failed checks on test: " + ", ".join(f"{k} x{v}" for k, v in sorted(failed.items())))
    text = "\n".join(lines) + "\n"
    _write(Path(cfg.run_dir) / "summary.txt", text)
    return text


def copy_library(src: str | Path, dst: str | Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True)
