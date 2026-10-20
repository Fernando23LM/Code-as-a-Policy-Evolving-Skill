# Self-evolving robot agents on lab bench tasks: session summary

10 to 14 October 2026, Gilbreth cluster. Charts are drawn from the run folders by
`reports/make_charts.py`.

## Summary

An LLM writes the code that controls two simulated YAM robot arms
(code-as-policy). It practises in rounds: each round it rewrites a skill
library, notes and a policy script, sees how they did on practice layouts, and
the best version is kept. Its final policy is then scored once on 20 held-out
layouts it never saw.

- **Three tasks of rising difficulty were built:** move a vial, open a
  snap-cap tube, and weigh out salt with both arms.
- **The weighing task separates the models.** The best result,
  `gpt-6-sol` learning the task in two halves with 24 practice layouts, passed
  18 of 20 held-out layouts on pouring alone and 15 of 20 on the full task. A
  hand-written reference passes 19. Claude Haiku 5.5 reached 0 to 9,
  `gpt-6-luna` 0 to 8 and `qwen3.8-27b` 0.
- **More practice layouts mattered most.** With 12 layouts per round, scores
  swung from round to round; with 24, `gpt-6-sol` climbed steadily and then
  held a perfect practice score for eight rounds.
- **Practice helps when the task is hard enough and selection is reliable.**
  On the snap-cap task, three practice runs scored 58/60 held-out against
  44/60 for fresh attempts without feedback.
- **Single runs mislead.** Identical setups ranged from 0 to 9 of 20, so every
  conclusion below rests on repeats where it says so.

## What was built

| Task | What the robot must do | Judge | Reference policy |
|---|---|---|---|
| `tube_rack` | right arm moves a named vial between racks | vial seated upright in the target cell | 40/40 |
| `snap_cap` | open the lid of a clamped tube past 90 degrees | lid angle, at rest, released | 64/64 |
| `weigh_salt` | left arm passes a salt vial to a shared rack; right arm pours onto a balance to a target mass (0.8 to 1.6 g, tolerance 0.2 g) | the balance reading itself | 61/64 seeds, 19/20 held-out |
| `weigh_handover` | first half of `weigh_salt`: vial into the shared rack | vial seated, released, no salt spilled | 20/20 |
| `weigh_pour` | second half: vial starts in the shared rack | the balance reading | 18 to 19/20 |

The two halves meet exactly, so `--compose` can run a learned hand-over script
and a learned pour script one after the other on the full task, with no model
call.

The reference policy on one layout of each, at 6x speed (made by `reports/make_gifs.py`):

| Move a vial | Open a snap cap | Hand the vial over (3a) |
|---|---|---|
| ![](gifs/tube_rack.gif) | ![](gifs/snap_cap.gif) | ![](gifs/weigh_handover.gif) |

| Pour to a target mass (3b) | Weigh out salt, both arms (3) |
|---|---|
| ![](gifs/weigh_pour.gif) | ![](gifs/weigh_salt.gif) |

Added to the practice loop (`biolab/evolve/`):

- **Partial credit** for failed layouts, used to break ties between rounds and
  shown to the model with the judge's reading.
- **Feedback** that names a vial left on the pan or on its side, and how far the
  vial slid in the gripper; the first and last part of each layout's output.
- **Fixed modules** (`--modules`) and **starting libraries** (`--inherit`) from
  earlier runs; **composition** (`--compose`) with no model.
- **A tolerant reply parser** (missing labels, unclosed or mangled code fences),
  an **output allowance that doubles** when a reasoning model returns nothing,
  `--reasoning-effort`, and **per-call token and cost logging**.
- **Per-round videos** with the balance reading on screen, and **layouts run in
  parallel**, one core each.

## Results on the weighing task

![Held-out score of every weighing run](weighing_held_out.png)

One dot per run; the number on the right is the total over the runs in that
row. "Joined" rows use Haiku's hand-over library (20/20) followed by the named
model's pour library.

| Setup | Model | Held-out per run (of 20) |
|---|---|---|
| One piece, no feedback | gpt-6-luna | 0 |
| One piece, library loop, empty start | gpt-6-luna | 0, 2 |
| One piece, starts from vial-moving skills | gpt-6-luna | 8, 0, 4, 0 |
| Same, plus grip hints | gpt-6-luna | 1, 0, 3, 1 |
| Same, plus grip hints | Claude Haiku 5.5 | 7, 0, 9, 3 |
| Two halves, pour on 12 layouts | qwen3.8-27b (low, medium effort) | 0, 0 |
| Two halves, pour on 12 layouts | Claude Haiku 5.5 | 1, 0, 4 |
| Two halves, pour on 12 layouts | gpt-6-luna | 6 |
| Two halves, pour on 12 layouts | gpt-6-sol | 13, 15 |
| Two halves, pour on 24 layouts | gpt-6-sol | 15 |

Pour-only held-out (of 20): reference 18 to 19; `gpt-6-sol` 13, 10 and 18
(24 layouts); Haiku 7, 4, 0, 1; luna 5; Qwen 0, 0.

![gpt-6-sol practice score by round](pour_practice_by_round.png)

The second 24-layout run stopped after round 4 when the provider account ran
out of credit; it was climbing like the first.

## Results on the two easier tasks

**Moving a vial** was too easy to show anything: `gpt-6-luna` passed 20/20
held-out after one model call, and fresh attempts without feedback did the same.

**Opening a snap cap** (held-out, of 20):

| Setup | Library loop | Fresh attempts, no feedback |
|---|---|---|
| 4 practice layouts | 10 | 14 |
| 12 practice layouts | 16 | 9 |
| 12 layouts, reworded lessons prompt, early stop | 13 | 16 |
| 12 layouts, no early stop, three repeats | 20, 20, 18 | 13, 16, 15 |

The early-stopping run froze a policy at its first perfect practice round,
which then failed on held-out layouts; letting practice continue fixed that.

## What we learned

- **Pouring is the hard part of the weighing task.** The hand-over was solved
  perfectly in a few rounds; every weaker model failed on dosing, on the vial
  slipping down the fingers, or on leaving the vial on the pan.
- **The loop's main contribution is keeping the best attempt.** Weaker models
  often got worse from one round to the next. Reliable selection needs enough
  practice layouts.
- **Learning in halves isolates the difficulty but cannot beat the harder
  half.** Joined scores tracked the pour score. The model did not use fixed
  modules when given the choice; it rewrote everything.
- **Hints about the grip made no measurable difference** for `gpt-6-luna`
  (4/60 with and 4/60 without over three repeats each).
- **Models need different call settings.** Haiku needed a larger output
  allowance; Qwen answered only with a reasoning-effort setting; `gpt-6-sol`
  worked with defaults.

## Cost and cluster use

- **Model spend:** a `gpt-6-sol` pour run cost $0.71 with 12 layouts and $1.36
  with 24 (feedback dominates the input). Qwen runs cost $0.15 and $0.42, luna
  runs about $0.06. The account ran out of credit during the last run.
- **Every job used standby QoS.** Requests were cut from 16 GB, 3 hours and any
  GPU partition to about 10 GB, under an hour and the A10/A30 partitions, after
  measuring peaks of 0.4 to 6.8 GB.
- **Two cluster problems were found and fixed:** jobs split across two nodes
  ran on a fraction of their cores (now `-N 1`), and the job script hid failures
  from Slurm (it now passes the exit status on).

## Open issues and next steps

- **Repeat the 24-layout `gpt-6-sol` run.** It stopped after round 4 for lack
  of credit; resubmitting resumes at round 5.
- **The joined score (15) trails pour-only (18).** Unverified guess: carrying
  the vial disturbs the salt the pour policy was tuned on.
- **Held-out tests and compositions are not filmed;** only practice rounds are.
- **Negative balance readings** in some failed layouts are still unexplained.
- **Housekeeping:** the Gilbreth copy of the code has moved well ahead of the
  PC copy and is not under version control; `biolab/evolve/evolve/` is a stray
  duplicate; development files sit in `biolab_runs/_dev`.
- **Toward a deployable robot:** use these scripts to collect camera
  demonstrations and fine-tune a vision-language-action model, starting with
  the hand-over, which already succeeds every time.

## Where everything is

- Code: `/scratch/gilbreth/flermama/biomedical-tasks` (`biolab/`, `cluster/run_evolve.sh`, `tests/`, `README.md`)
- Runs, with per-round prompts, replies, results and videos: `/scratch/gilbreth/flermama/biolab_runs/<run name>`
- This report and its charts: `reports/` in the code folder
- Job logs: `evolve_<job id>.log` in the code folder
