"""Charts for reports/session_summary.md, read straight from the run folders.

    python reports/make_charts.py [runs folder]
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/gilbreth/flermama/biolab_runs")
OUT = Path(__file__).resolve().parent
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODEL = {"gpt-6-luna": "#2a78d6", "Claude Haiku 5.5": "#eb6834", "qwen3.8-27b": "#1baf7a", "gpt-6-sol": "#eda100"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2})


def held_out(run):
    path = RUNS / run / "test.json"
    return json.loads(path.read_text())["successes"] if path.exists() else None


def practice(run):
    rounds = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(RUNS / run / "round_*/results.json")))]
    return [r["successes"] / r["episodes"] for r in rounds]


# (group, row label, model, runs)
ROWS = [
    ("Full task learned in one piece", "no feedback between rounds", "gpt-6-luna", ["luna_weigh_restarts"]),
    ("Full task learned in one piece", "library loop, empty start", "gpt-6-luna", ["luna_weigh", "luna_weigh_v2"]),
    ("Full task learned in one piece", "library loop, starts from vial-moving skills", "gpt-6-luna",
     ["luna_weigh_v3_inherit", "luna_weigh_cmp_plain_1", "luna_weigh_cmp_plain_2", "luna_weigh_cmp_plain_3"]),
    ("Full task learned in one piece", "same, plus grip hints", "gpt-6-luna",
     ["luna_weigh_v4_inherit", "luna_weigh_cmp_hints_1", "luna_weigh_cmp_hints_2", "luna_weigh_cmp_hints_3"]),
    ("Full task learned in one piece", "same, plus grip hints", "Claude Haiku 5.5",
     ["haiku_weigh_inherit", "haiku_weigh_rep_1", "haiku_weigh_rep_2", "haiku_weigh_rep_3"]),
    ("Learned in two halves, then joined", "pour learned on 12 layouts", "qwen3.8-27b", ["qwen_seq_composed", "qwen_medium_composed"]),
    ("Learned in two halves, then joined", "pour learned on 12 layouts", "Claude Haiku 5.5",
     ["haiku_seq_composed_1", "haiku_seq_composed_2", "haiku_seq_composed_3"]),
    ("Learned in two halves, then joined", "pour learned on 12 layouts", "gpt-6-luna", ["luna_seq_composed"]),
    ("Learned in two halves, then joined", "pour learned on 12 layouts", "gpt-6-sol", ["sol_seq_composed", "sol_seq_composed_2"]),
    ("Learned in two halves, then joined", "pour learned on 24 layouts", "gpt-6-sol", ["sol_seq_composed_24", "sol_seq_composed_24_b"]),
]


def chart_held_out():
    fig, ax = plt.subplots(figsize=(9.2, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y, ticks, labels, last_group, table = 0.0, [], [], None, []
    for group, label, model, runs in ROWS:
        if group != last_group:
            y -= 0.9 if last_group else 0.0
            ax.text(-0.6, y + 0.72, group, fontsize=10.5, fontweight="bold", color=INK, ha="left", va="center")
            last_group = group
        scores = [s for s in (held_out(r) for r in runs) if s is not None]
        seen = {}
        for s in sorted(scores):                      # stack equal scores so every run stays visible
            k = seen.get(s, 0); seen[s] = k + 1
            ax.scatter(s, y + 0.30 * k - 0.15 * (scores.count(s) - 1), s=70, color=MODEL[model], edgecolor=SURFACE,
                       linewidth=1.5, zorder=3)
        if scores:
            ax.text(20.7, y, f"{sum(scores)}/{20 * len(scores)}", fontsize=9, color=INK2, va="center")
        ticks.append(y); labels.append(f"{model}: {label}")
        table.append((group, model, label, scores))
        y -= 1.0
    ax.axvline(19, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    ax.text(18.8, 1.15, "hand-written reference: 19", fontsize=8.5, color=MUTED, ha="right", va="center")
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(-0.8, 22.6); ax.set_ylim(y + 0.3, 1.9)
    ax.set_xticks(range(0, 21, 5)); ax.set_xlabel("held-out layouts passed, out of 20 (one dot per run)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for side in ("top", "right", "left"): ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=c, label=m) for m, c in MODEL.items()]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=4, frameon=False, fontsize=9,
              labelcolor=INK2, handletextpad=0.2, columnspacing=1.4)
    ax.set_title("Weighing task: held-out score of every run", loc="left", fontsize=13, fontweight="bold", pad=34)
    fig.tight_layout(); fig.savefig(OUT / "weighing_held_out.png", dpi=170, facecolor=SURFACE); plt.close(fig)
    return table


def chart_practice():
    series = [("12 layouts, run 1", "sol_seq_pour", "#86b6ef", "-"), ("12 layouts, run 2", "sol_seq_pour_2", "#86b6ef", (0, (4, 2))),
              ("24 layouts, run 1", "sol_seq_pour_24", "#184f95", "-"), ("24 layouts, run 2 (stopped after round 4)", "sol_seq_pour_24_b", "#184f95", (0, (4, 2)))]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for label, run, color, style in series:
        ys = practice(run)
        if not ys: continue
        ax.plot(range(len(ys)), [100 * v for v in ys], color=color, linewidth=2, linestyle=style, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1, label=label, zorder=3)
    ax.set_xlim(-0.3, 11.3); ax.set_ylim(-4, 106)
    ax.set_xticks(range(12)); ax.set_yticks(range(0, 101, 25)); ax.set_yticklabels([f"{v}%" for v in range(0, 101, 25)])
    ax.set_xlabel("practice round"); ax.set_ylabel("practice layouts passed")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), frameon=False, fontsize=9, labelcolor=INK2, ncol=2,
              columnspacing=1.6)
    ax.set_title("gpt-6-sol on the pour task: practice score by round", loc="left", fontsize=13, fontweight="bold", pad=46)
    fig.tight_layout(); fig.savefig(OUT / "pour_practice_by_round.png", dpi=170, facecolor=SURFACE); plt.close(fig)


if __name__ == "__main__":
    for group, model, label, scores in chart_held_out():
        print(f"{group} | {model} | {label} | {scores}")
    chart_practice()
    for name in ("sol_seq_pour", "sol_seq_pour_2", "sol_seq_pour_24", "sol_seq_pour_24_b", "haiku_seq_pour_1", "haiku_seq_pour_2", "haiku_seq_pour_3",
                 "haiku_stage_pour", "luna_seq_pour", "qwen_seq_pour", "qwen_medium_pour", "haiku_seq_handover", "haiku_stage_full"):
        print(name, "held-out", held_out(name))
