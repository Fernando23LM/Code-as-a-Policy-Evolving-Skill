"""Command line for the library loop.

    python -m biolab.evolve --run-dir runs/luna_a --model gpt-6-luna \\
        --base-url https://example.com/v1/ --key-file ~/.key --rounds 8
    python -m biolab.evolve --run-dir runs/luna_a --test-only
    python -m biolab.evolve --run-dir runs/check --reference        # no model: scores the hand-written policy
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..api import REFERENCE_POLICY
from .llm import make_llm
from .loop import Config, _load_rounds, _write, practice, summarize, test


def _seed_range(text: str) -> list[int]:
    """'100-103' or '100,101,105'."""
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description="Practice with a persistent skill library, then test on held-out layouts.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1/")
    ap.add_argument("--key-file")
    ap.add_argument("--rounds", type=int, default=8, help="practice budget in model calls")
    ap.add_argument("--practice-seeds", type=_seed_range, default=_seed_range("100-103"))
    ap.add_argument("--test-seeds", type=_seed_range, default=_seed_range("1000-1019"))
    ap.add_argument("--inherit", help="library folder of an earlier run to start from")
    ap.add_argument("--restarts", action="store_true",
                    help="baseline: every round is a fresh attempt with no library and no feedback")
    ap.add_argument("--no-early-stop", action="store_true", help="keep practising after a perfect round")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--test-only", action="store_true", help="skip practice; score the saved library")
    ap.add_argument("--reference", action="store_true", help="score the hand-written reference policy instead")
    args = ap.parse_args()

    cfg = Config(run_dir=args.run_dir, rounds=args.rounds, practice_seeds=args.practice_seeds,
                 test_seeds=args.test_seeds, inherit=args.inherit, memory=not args.restarts,
                 stop_when_perfect=not args.no_early_stop, model=args.model)
    if args.reference:
        cfg.model = "hand-written reference"
        for name, text in (("skills.py", ""), ("lessons.md", ""), ("policy.py", REFERENCE_POLICY)):
            _write(Path(cfg.run_dir) / "library" / name, text)
        rounds = []
    elif args.test_only:
        rounds = _load_rounds(Path(cfg.run_dir))
    else:
        if not (args.model and args.key_file):
            ap.error("--model and --key-file are required unless --test-only or --reference is given")
        llm = make_llm(args.base_url, Path(args.key_file).expanduser(), args.model, args.temperature)
        rounds = practice(cfg, llm)
    print()
    print(summarize(cfg, rounds, test(cfg)))


if __name__ == "__main__":
    main()
