"""Charts for reports/session_summary.md, read straight from the run folders.

    python reports/make_charts.py [runs folder]
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
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
              ("24 layouts", "sol_seq_pour_24", "#184f95", "-")]   # the 24-layout repeat stopped after round 4 is left out
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for (label, run, color, style), mark in zip(series, ("o", "s", "^")):
        ys = practice(run)
        if not ys: continue
        ax.plot(range(len(ys)), [100 * v for v in ys], color=color, linewidth=2, linestyle=style, marker=mark, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1, label=label, zorder=3)
    ax.set_xlim(-0.3, 11.3); ax.set_ylim(-4, 106)
    ax.set_xticks(range(12)); ax.set_yticks(range(0, 101, 25)); ax.set_yticklabels([f"{v}%" for v in range(0, 101, 25)])
    ax.set_xlabel("practice round"); ax.set_ylabel("practice layouts passed")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), frameon=False, fontsize=9, labelcolor=INK2, ncol=3,
              handlelength=4.5,
              columnspacing=1.6)
    ax.set_title("gpt-6-sol on the pour task: practice score by round", loc="left", fontsize=13, fontweight="bold", pad=46)
    fig.tight_layout(); fig.savefig(OUT / "pour_practice_by_round.png", dpi=170, facecolor=SURFACE); plt.close(fig)


def chart_models_by_round():
    """Pour task, practice score per round: one panel per model, one line per run."""
    panels = [
        ("gpt-6-sol", [("12 layouts, run 1", "sol_seq_pour"), ("12 layouts, run 2", "sol_seq_pour_2"),
                       ("24 layouts", "sol_seq_pour_24")]),
        ("gpt-6-luna", [("12 layouts", "luna_seq_pour")]),
        ("Claude Haiku 5.5", [("12 layouts, run 1", "haiku_seq_pour_1"), ("12 layouts, run 2", "haiku_seq_pour_2"),
                              ("12 layouts, run 3", "haiku_seq_pour_3"), ("12 layouts, 8 rounds", "haiku_stage_pour")]),
    ]
    styles = ["-", (0, (5, 2)), (0, (1.5, 1.5)), (0, (6, 2, 1.5, 2))]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.6), sharey=True, facecolor=SURFACE)
    for ax, (model, runs) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for (label, run), style in zip(runs, styles):
            ys = practice(run)
            if not ys:
                continue
            ax.plot(range(len(ys)), [100 * v for v in ys], color=MODEL[model], linewidth=2, linestyle=style,
                    marker="o", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1, zorder=3,
                    label=f"{label} (held-out {held_out(run)}/20)")
        ax.set_title(model, loc="left", fontsize=11.5, fontweight="bold", color=INK)
        ax.set_xlim(-0.4, 11.4); ax.set_ylim(-4, 106)
        ax.set_xticks(range(0, 12, 2)); ax.set_xlabel("practice round")
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), frameon=False, fontsize=8.5, labelcolor=INK2,
                  handlelength=2.8)
    axes[0].set_yticks(range(0, 101, 25)); axes[0].set_yticklabels([f"{v}%" for v in range(0, 101, 25)])
    axes[0].set_ylabel("practice layouts passed")
    fig.suptitle("Pour task: practice score by round, every run", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "pour_practice_by_model.png", dpi=170, facecolor=SURFACE); plt.close(fig)


POUR_RUNS = {   # model -> [(run folder, label)] on the pour task; the stopped sol_seq_pour_24_b is left out
    "gpt-6-sol": [("sol_seq_pour", "run 1, 12 layouts"), ("sol_seq_pour_2", "run 2, 12 layouts"),
                  ("sol_seq_pour_24", "run 3, 24 layouts")],
    "gpt-6-luna": [("luna_seq_pour", "run 1, 12 layouts")],
    "Claude Haiku 5.5": [("haiku_stage_pour", "run 1, 12 layouts"), ("haiku_seq_pour_1", "run 2, 12 layouts"),
                         ("haiku_seq_pour_2", "run 3, 12 layouts"), ("haiku_seq_pour_3", "run 4, 12 layouts")],
}
STYLES = ["-", (0, (6, 3)), (0, (1, 2)), (0, (7, 2, 1.5, 2))]
MARKERS = ["o", "s", "^", "D"]   # a different marker per run, so runs differ even in a short legend sample


def chart_pour_rounds(models=tuple(POUR_RUNS), name="pour_rounds_by_model.png"):
    """Practice score per round and held-out score, one panel per model, on the pour task."""
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models) + 0.3 if len(models) > 1 else 7.4, 5.6), sharey=True, facecolor=SURFACE,
                             squeeze=False)
    axes = axes[0]
    test_x = 13.2
    for ax, model in zip(axes, models):
        runs = POUR_RUNS[model]
        ax.set_facecolor(SURFACE)
        color = MODEL[model]
        for i, ((run, label), style, mark) in enumerate(zip(runs, STYLES, MARKERS)):
            ys = [100 * v for v in practice(run)]
            test = held_out(run)
            line, = ax.plot(range(len(ys)), ys, color=color, linewidth=2, linestyle=style, marker=mark, markersize=5,
                            markeredgecolor=SURFACE, markeredgewidth=0.8, label=f"{label}: held-out {test}/20", zorder=3)
            ax.scatter(test_x, 5 * test, marker="D", s=60, color=color, edgecolor=INK, linewidth=0.8, zorder=4)
            ax.text(test_x + 0.45, 5 * test, f"run {i + 1}", fontsize=8, color=INK2, va="center")
        ax.axvline(12.1, color=AXIS, linewidth=1)
        ax.set_xlim(-0.4, 15.2); ax.set_ylim(-4, 106)
        ax.set_xticks(list(range(0, 12, 2)) + [test_x]); ax.set_xticklabels([str(i) for i in range(0, 12, 2)] + ["held-out"])
        ax.set_yticks(range(0, 101, 25)); ax.set_yticklabels([f"{v}%" for v in range(0, 101, 25)])
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"): ax.spines[side].set_visible(False)
        ax.set_xlabel("practice round")
        ax.set_title(model, loc="left", fontsize=11.5, fontweight="bold", color=INK)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), frameon=False, fontsize=8.5, labelcolor=INK2,
                  handlelength=5.0)
    axes[0].set_ylabel("layouts passed")
    fig.suptitle("Pour task: practice score by round, and the kept policy on 20 held-out layouts (diamonds)"
                 if len(models) > 1 else "Pour task: practice score by round, held-out score (diamonds)",
                 x=0.01, ha="left", fontsize=13 if len(models) > 1 else 12.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / name, dpi=170, facecolor=SURFACE); plt.close(fig)


DRUG_RUNS = {   # model -> (run with feedback, no-feedback run of three independent attempts)
    "gpt-6-sol": ("drug_sol_24", "drug_nofeedback_sol"),
    "gpt-6-luna": ("drug_luna_24", "drug_nofeedback_luna"),
}


def chart_drug_feedback():
    """Drug-response task: practice by round with feedback, three no-feedback attempts, and both held-out scores."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True, facecolor=SURFACE)
    test_x = 13.4
    for ax, (model, (fed, unfed)) in zip(axes, DRUG_RUNS.items()):
        ax.set_facecolor(SURFACE)
        color = MODEL[model]
        ys = [100 * v for v in practice(fed)]
        ax.plot(range(len(ys)), ys, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=0.8, zorder=3, label=f"with feedback, 12 rounds: held-out {held_out(fed)}/20")
        tries = [100 * v for v in practice(unfed)]
        ax.scatter(range(len(tries)), tries, s=55, facecolor=SURFACE, edgecolor=INK2, linewidth=1.6, zorder=4,
                   label=f"no feedback, 3 separate attempts: held-out {held_out(unfed)}/20")
        # Side by side, so equal scores do not hide each other; each labelled just below its diamond.
        for dx, value, filled, text in ((-0.8, held_out(fed), True, "with\nfeedback"), (0.8, held_out(unfed), False, "no\nfeedback")):
            ax.scatter(test_x + dx, 5 * value, marker="D", s=70, facecolor=color if filled else SURFACE,
                       edgecolor=INK if filled else INK2, linewidth=0.8 if filled else 1.6, zorder=5)
            ax.text(test_x + dx, 5 * value - 5, text, fontsize=7.5, color=INK2, ha="center", va="top")
        ax.axvline(12.1, color=AXIS, linewidth=1)
        ax.set_xlim(-0.4, 16.4); ax.set_ylim(-4, 106)
        ax.set_xticks(list(range(0, 12, 2)) + [test_x]); ax.set_xticklabels([str(i) for i in range(0, 12, 2)] + ["held-out"])
        ax.set_yticks(range(0, 101, 25)); ax.set_yticklabels([f"{v}%" for v in range(0, 101, 25)])
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"): ax.spines[side].set_visible(False)
        ax.set_xlabel("practice round (24 layouts each)")
        ax.set_title(model, loc="left", fontsize=11.5, fontweight="bold", color=INK)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), frameon=False, fontsize=8.5, labelcolor=INK2,
                  handlelength=3.5)
    axes[0].set_ylabel("layouts passed")
    fig.suptitle("Drug dose-response task: practice with feedback against no feedback, and held-out score (diamonds)",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "drug_feedback_vs_none.png", dpi=170, facecolor=SURFACE); plt.close(fig)


def credit(run):
    """Mean partial credit per round (0 to 1)."""
    rounds = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(RUNS / run / "round_*/results.json")))]
    return [float(r.get("progress") or 0.0) for r in rounds]


def chart_luna_lab_tasks(with_skills=False, name="luna_protein_culture.png"):
    """gpt-6-luna on the protein assay and the cell culture: feedback against no feedback, with partial credit.

    with_skills adds the culture run that started from luna's protein-assay library.
    """
    panels = {"Protein assay": ("protein_assay_luna_24", "protein_assay_nofeedback_luna", None),
              "Cell culture (one simulated week)": ("cell_culture_luna_24", "cell_culture_nofeedback_luna",
                                                    "cell_culture_luna_24_inherit_protein" if with_skills else None)}
    color, dark = MODEL["gpt-6-luna"], "#104281"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.9 if with_skills else 5.6), sharey=True, facecolor=SURFACE)
    test_x = 13.4
    for ax, (title, (fed, unfed, skilled)) in zip(axes, panels.items()):
        ax.set_facecolor(SURFACE)
        start = "empty start, " if skilled else ""
        ys = [100 * v for v in practice(fed)]
        ax.plot(range(len(ys)), ys, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=0.8, zorder=3, label=f"with feedback, {start}layouts passed (held-out {held_out(fed)}/20)")
        ax.plot(range(len(ys)), [100 * v for v in credit(fed)], color=color, linewidth=1.3, linestyle=(0, (2, 2)),
                alpha=0.8, zorder=2, label=f"with feedback, {start}partial credit")
        if skilled:
            ks = [100 * v for v in practice(skilled)]
            ax.plot(range(len(ks)), ks, color=dark, linewidth=2.2, marker="s", markersize=5, markeredgecolor=SURFACE,
                    markeredgewidth=0.8, zorder=4,
                    label=f"with feedback, starting from its protein-assay skills (held-out {held_out(skilled)}/20)")
        tries = [100 * v for v in practice(unfed)]
        ax.scatter(range(len(tries)), tries, s=55, facecolor=SURFACE, edgecolor=INK2, linewidth=1.6, zorder=5,
                   label=f"no feedback, 3 separate attempts (held-out {held_out(unfed)}/20)")
        marks = [(held_out(fed), color, INK, 0.8, "o", "empty\nstart" if skilled else "with\nfeedback")]
        if skilled:
            marks.append((held_out(skilled), dark, INK, 0.8, "s", "protein\nskills"))
        marks.append((held_out(unfed), SURFACE, INK2, 1.6, "o", "no\nfeedback"))
        offsets = np.linspace(-1.1, 1.1, len(marks)) if len(marks) > 2 else [-0.8, 0.8]
        for dx, (value, face, edge, width, _, text) in zip(offsets, marks):
            ax.scatter(test_x + dx, 5 * value, marker="D", s=70, facecolor=face, edgecolor=edge, linewidth=width, zorder=6)
            ax.text(test_x + dx, 5 * value + (7 if value == 0 else -5), text, fontsize=7.2, color=INK2, ha="center",
                    va="bottom" if value == 0 else "top")
        ax.axvline(12.1, color=AXIS, linewidth=1)
        ax.set_xlim(-0.4, 16.6); ax.set_ylim(-4, 106)
        ax.set_xticks(list(range(0, 12, 2)) + [test_x]); ax.set_xticklabels([str(i) for i in range(0, 12, 2)] + ["held-out"])
        ax.set_yticks(range(0, 101, 25)); ax.set_yticklabels([f"{v}%" for v in range(0, 101, 25)])
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"): ax.spines[side].set_visible(False)
        ax.set_xlabel("practice round (24 layouts each)")
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), frameon=False, fontsize=8.2, labelcolor=INK2,
                  handlelength=3.5)
    axes[0].set_ylabel("layouts passed, or partial credit")
    fig.suptitle("gpt-6-luna on two lab tasks: practice with feedback against no feedback, and held-out score (diamonds)",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / name, dpi=170, facecolor=SURFACE); plt.close(fig)


if __name__ == "__main__":
    for group, model, label, scores in chart_held_out():
        print(f"{group} | {model} | {label} | {scores}")
    chart_practice()
    chart_pour_rounds()
    chart_drug_feedback()
    chart_luna_lab_tasks()
    chart_luna_lab_tasks(with_skills=True, name="luna_protein_culture_skills.png")
    chart_pour_rounds(("gpt-6-sol",), "pour_rounds_gpt-6-sol.png")
    chart_models_by_round()
    for name in ("sol_seq_pour", "sol_seq_pour_2", "sol_seq_pour_24", "sol_seq_pour_24_b", "haiku_seq_pour_1", "haiku_seq_pour_2", "haiku_seq_pour_3",
                 "haiku_stage_pour", "luna_seq_pour", "qwen_seq_pour", "qwen_medium_pour", "haiku_seq_handover", "haiku_stage_full"):
        print(name, "held-out", held_out(name))
