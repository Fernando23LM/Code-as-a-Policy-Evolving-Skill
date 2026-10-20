# Self-evolving robot agents on lab bench tasks: Does the skill library improve performance?

## Introduction 

Inspired by [Self Evolving Agentic Robotics](https://sear.bot) paper, an LLM writes the code that controls two simulated YAM robot arms
(code-as-policy). It practises in rounds: each round it rewrites a skill
library, notes and a policy script, sees how they did on practice layouts, and
the best version is kept. Its final policy is then scored once on 20 held-out
layouts it never saw.


## Tasks

| Task | What the robot must do | Judge | Reference policy |
|---|---|---|---|
| `tube_rack` ![](/reports/Move_vial_between_racks.gif) | right arm moves a named vial between racks | Vial seated upright in the target cell | 40/40 |
| `snap_cap` ![](/reports/Open_snap_cap_tube.gif) | Open the lid of a clamped tube past 90 degrees | Lid angle, at rest, released | 64/64 |
| `weigh_salt` ![](/reports/Pour_salt.gif) | left arm passes a salt vial to a shared rack; right arm pours onto a balance to a target mass (0.8 to 1.6 g, tolerance 0.2 g) | The balance reading itself | 61/64 seeds, 19/20 held-out |
| `weigh_handover` ![](/reports/Move_vial_to_middle.gif) | first half of `weigh_salt`: vial into the shared rack | Vial seated, released, no salt spilled | 20/20 |
| `weigh_pour` ![](/reports/Pour_salt_from_middle.gif) | second half: vial starts in the shared rack | The balance reading | 18 to 19/20 |
| `cell_culture` ![](/reports/cell_culture.gif) | feeding cells, splitting dish A into B and C, and growth to 2.4 million cells against a target of 1.8 million. Another clock simulates time and cell growth | The three dishes have at least 1.8 million attached cells<br>At least 90% of the cells are alive<br>No dish spent more than 6 hours less than 15% of nutrients<br>No dish spent more than 12 hours above 95%<br>Every dish holds at least 1 mL at the end| 18/20 |
| `protein_assay` ![](/reports/protein_assay.gif) | find protein concentration in an unknown liquid by making a set of reference liquids with known amounts, measuring their colour with an instrument, and reading the unknown off that colour scale.| Value of 6 references by 10% <br> Sample contains 100 µL <br> Concentration reported within 10% of correct value| 20/20 |
| `drug_response` | 1. Seed cancer cells into a plate. Add a dilution series of a drug, using the protein assay’s pipetting.<br>2. Incubate for two days, using the culture’s growth model, with the drug slowing growth or killing cells depending on dose.<br>3. Read viability on the plate reader and report the dose that halves growth (the IC50).| The robot passes if its estimate of the drug concentration that halves cell growth is within 50% of the true value <br> Concentration reported within 10% of correct value| 20/20 |

## Workflow

### 1. Starting a run
A round contains several runs. On each round there are several layouts. 

`RUN=sol_seq_pour MODEL=gpt-6-sol TASK=weigh_pour ROUNDS=12 EXTRA="--practice-seeds 100-123 --video" sbatch --qos=standby -N 1 -n 12 --mem=10G --time=00:40:00 cluster/run_evolve.sh`

### 2. Each round
For each round:

1. **Build prompt**: contains task description, API documentation and current best library. Input to the LLM model

2. **Call the model once**: it returns a skills module, a lessons file and a policy script

3. **Parse reply**: if no policy script is found, the round score is zero and is discarded

4. **Run the code each layout**: each layout is a different random arrangement of the bench, the same set every round. We tested with 12 or 24 layouts per round

5. **Keep or discard**: if the round passes as many layouts as the previous library it becomes the new library, if not the library does not change.

Everything is saved on `round_NNN`

### 3. Held-out test
After the last round, the best library is tested against 20 layouts it has not seen. Score is in `test.json` and `summary.txt`


## Results on the weighing task

![Held-out score of every weighing run](/reports/Final-data.png)

Score achieved on each round, the diamonds show the score achieved once the held-out test on the 12 rounds finished. Score increased when switching to a better LLM model (from Haiku and Luna to Sol) and when number of layouts increased (12 to 24).


## Results on the two easier tasks

**Moving a vial** was too easy to show anything: `gpt-6-luna` passed 20/20
held-out after one model call, and fresh attempts without feedback did the same.

**Opening a snap cap** left columns show the held-out score with the library loop and with no feedback

| Setup | Library loop | Fresh attempts, no feedback |
|---|---|---|
| 4 practice layouts | 10/20 | 14/20 |
| 12 practice layouts | 16/20 | 9/20 |
| 12 layouts, reworded lessons prompt, early stop | 13/20 | 16/20 |
| 12 layouts, no early stop, three repeats | 20/20, 20/20, 18/20 | 13/20, 16/20, 15/20 |


## Results on the cell, protein and drug task

![Protein Assay & Cell Culture tasks](/reports/protein_assay_cell_culture.png)
For the protein assay gpt-6-luna could wrote a correct script and achieved maximum score quickly.
For the cell culture tasks, which involves handling pipettes, the score improved if the model could use the protein assay library. This suggests that previous library skills can be used as the starting point for more difficult tasks.<br>
**Regarding the score drops from round 4 to 5 and 8 to 9:** On round 4 the model ran out of time several layouts, so on round 5 it avoided splitting dishes and failed to reach the required amount of cells. For round 8, some cells dishes starved while having maximum liquid volume. The important metric is nutrient concentration, not volume. On round 9 model replaced completely the liquid when there was no more nutrient, which takes longer, so the robot ran out of time. On round 10 it found the balance.


![Drug response](/reports/drug-response.png)
  With 24 layouts per round gpt-6-luna reached the same score as gpt-6-sol. Luna being signficantly cheaper than sol. 

## Key findings
- **Skill library improves with more layouts per round**
- **Previous Library skills can be reused** on the cell culture, score increased if the library skill for protein assay was given

- **Pouring is the hard part of the weighing task.** The hand-over was solved
  perfectly in a few rounds; every weaker model failed on dosing, on the vial
  slipping down the fingers, or on leaving the vial on the pan.

- **The loop's main contribution is keeping the best attempt.** Weaker models
  often got worse from one round to the next. Reliable selection needs more
  practice layouts.
- **Learning in halves isolates the difficulty but cannot beat the harder
  half.** Joined scores tracked the pour score. The model did not use fixed
  modules when given the choice; it rewrote everything.
- **Models need different call settings.** Haiku needed a larger output
  allowance; Qwen answered only with a reasoning-effort setting; `gpt-6-sol`
  worked with defaults.


## Open issues and next steps

- **Repeat with 24-layouts or more** runnings with 12 layouts do not improve the skill library 
- **Remove ideal robot information:** model gets robot position, not image. 
- **Held-out tests and compositions are not filmed;** only practice rounds are.
- **Negative balance readings** in some failed layouts are still unexplained.
- **Toward a deployable robot:** use these scripts to collect camera
  demonstrations and fine-tune a vision-language-action model, starting with
  the hand-over, which already succeeds every time.

## File Structure

- Code:  `biolab/`, `cluster/run_evolve.sh`, `tests/`, `README.md`
- Runs, with per-round prompts, replies, results and videos: `/biolab_runs/<run name>`
- This report and its charts: `reports/` in the code folder
- Job logs: `evolve_<job id>.log` in the code folder



# How to use


```
biolab/scene.py     builds the MuJoCo model (arms, racks, vials, cameras)
biolab/env.py       TubeRackEnv: reset, perceive, move_to, set_gripper, render; SnapCapEnv
biolab/judge.py     success checks (keep out of the agent's view)
biolab/api.py       the functions a model may call, and the reference policies
scripts/scripted_policy.py   hand-written baseline, 40/40 on seeds 0-39 (MuJoCo 3.15 and 3.5)
tests/              pytest: stability, determinism, baseline, unreachable target, snap-cap and weighing tasks
previews/           before/after images from seed 0
third_party/        YAM model from MuJoCo Menagerie (see its LICENSE)
```

## Run

```
.venv\Scripts\python scripts\scripted_policy.py --seeds 10
.venv\Scripts\python -m pytest -q tests
```

On Linux: `python -m venv .venv && .venv/bin/pip install -r requirements.txt`.
Physics needs no GPU. `env.render()` needs an OpenGL context; on a headless
node set `MUJOCO_GL=egl` 

## Policy API

```python
env = TubeRackEnv(seed=0)
env.task.instruction                       # "Move tube_0 from the source rack into cell (2, 1) ..."
obs = env.perceive()                       # tubes (bottom, top, axis), rack cells and yaw, gripper poses
env.move_to("right", [x, y, z], yaw, tilt) # straight-line move of the pinch point; False if unreachable
env.set_gripper("right", 0.05)             # distance between the finger pads, metres
env.home("right")
img = env.render("overhead")               # also "front", "left/wrist_cam", "right/wrist_cam"
env.evaluate()                             # judge: {"success", "checks", ...}
```

## CaP-X (LLM-written policies)

`biolab/capx_env/` wraps the scene for CaP-X (https://github.com/capgym/cap-x)
without patching it: the YAML configs in `env_configs/tube_rack/` name our task
class, and importing it registers the simulator, the API and the task.

```
biolab/capx_env/sim.py    TubeRackLowLevel: reset/observe/reward, video frames, 3-minute sim budget
biolab/capx_env/api.py    the functions the model may call; its docstrings are the model's only manual
biolab/capx_env/task.py   prompt and reference ("oracle") solution
cluster/install_capx.sh       installs CaP-X + this project under $capx
cluster/run_capx_oracle.sh    reference solutions through CaP-X's launcher, no LLM
cluster/run_capx_llm.sh       an LLM writes the policy (needs an OpenRouter key)
```

With this folder copied to `biomedical-tasks`:

```
cd biomedical-tasks
sbatch cluster/install_capx.sh       # once; look for "INSTALL OK" in capx_install_<job>.log
sbatch cluster/run_capx_oracle.sh    # expect the tube-rack task to pass 10/10
sbatch cluster/run_capx_llm.sh       # after putting the key in $capx/cap-x/.openrouterkey
```

Status: on this PC the reference solution passes through CaP-X's own
`CodeExecutionEnvBase` and config loader with MuJoCo 3.5.0 (the version CaP-X
locks).
Known gaps:
- CaP-X hands the generated code the raw `env` object, so a model could read
  the judge through `env.sim`. Fine for getting started, not for reported numbers.
- CaP-X seeds each trial with its index, so every run sees layouts 0..N-1.
  `TubeRackLowLevel(seed_offset=...)` exists for a practice/test split but no
  config uses it yet.
- Only the ground-truth ("privileged") API exists. Camera images, depth,
  intrinsics and poses are already in the observation for a later vision API.

## Library loop and the second task

`biolab/evolve/` is the practice loop: each round the model rewrites a skill
library, lessons and a policy, the policy runs on practice layouts (seeds
100-103), and the best-scoring library is kept. The saved policy is then scored
once on held-out layouts (seeds 1000-1019). `biolab/tasks.py` lists the tasks.

```
RUN=luna_loop MODEL=gpt-6-luna sbatch cluster/run_evolve.sh                  # task 1: move a vial
RUN=luna_cap  MODEL=gpt-6-luna TASK=snap_cap sbatch cluster/run_evolve.sh    # task 2: open a snap cap
RUN=reference_cap TASK=snap_cap EXTRA="--reference" sbatch cluster/run_evolve.sh   # no model, no cost
```

Add `EXTRA="--practice-seeds 100-111 --video"` for more practice layouts and a
`practice.mp4` in every round folder (all practice layouts back to back, 4x speed).

Task 1 (`tube_rack`) is solved by gpt-6-luna on the first call, 20/20 held-out,
with or without the library, so it cannot show whether practice helps.

Task 2 (`snap_cap`, `SnapCapEnv`): two snap-cap tubes in one rack, each turned
its own way; open the named lid to at least 90 degrees, leave it at rest and
let go, with the other lid still shut. The reference policy pinches the lid
across its width, follows its arc to about upright, lets go and pushes it back
to its stop; it opens 64/64 (seeds 0-39, 100-103 and 1000-1019).

- The tubes have no joint: the rack clamps them. A free tube would be lifted
  out by its lid, and holding it down needs the left arm, which cannot reach.
- The latch is a closing torque (0.09 N m) that fades to zero at 0.12 rad and
  is zero beyond. Past it the lid stays wherever it is left.
- What makes it harder than task 1: the gripper yaw must be folded into
  +/-90 degrees or `move_to` refuses; pulling straight up stalls near 40
  degrees; the lid slips in the pads and lags the gripper by about 20 degrees,
  so an open-loop arc to 90 ends short; `move_to` often returns False while
  the lid is held; and `go_home` from low over the rack knocks the lid shut.
- No CaP-X binding for this task yet.

## Task 3: weigh out salt with both arms

`weigh_salt` (`WeighEnv`): an open vial of coarse salt starts in a rack only the
left arm reaches; a balance with a tared weigh boat stands where only the right
arm reaches; a small rack between them is the one place both reach. The target
is 0.8 to 1.6 g of the 2.4 g in the vial, and the judge is the balance: its
reading within 0.2 g of the target, steady, with no gripper or vial on the pan.

```
RUN=reference_weigh TASK=weigh_salt EXTRA="--reference" sbatch -n 16 --mem=12G cluster/run_evolve.sh
RUN=luna_weigh MODEL=gpt-6-luna TASK=weigh_salt EXTRA="--practice-seeds 100-111 --video --no-early-stop" \
    sbatch -n 12 --mem=12G cluster/run_evolve.sh
```

A layout takes about 95 simulated seconds and 70 s of wall time, so `--workers`
(set from `-n` by the job script) runs layouts side by side, one core each.

- The reference policy hands the vial over through the middle rack, leans it
  over the boat with `tilt`, creeps until the reading nears the target, and
  puts the vial back. It passes 61/64 on seeds 0-63 and 19/20 held-out; every
  miss is an overshoot of 0.2 to 0.25 g from one last avalanche of grains.
- Salt is 68 spheres of mixed size (2.8-3.6 mm, 0.02-0.05 g). Equal spheres
  locked into layers and left ten at a time. The cup in the vial top flares by
  20 degrees so it pours at 60-75 degrees of lean, inside the arm's reach.
- The balance is a pan on a stiff spring; the reading is its sag, averaged
  over half a second and tared at reset. The boat is part of the pan.
- This scene uses MuJoCo's CG solver (Newton was 14 times slower with the
  grains). Tasks 1 and 2 are unchanged.
- The `wait(seconds)` API function exists only here.
- Failed layouts earn partial credit (0.1 vial out of the stock rack, 0.1 any
  salt on a clear pan, 0.8 closeness of the reading); it breaks ties between
  rounds and is shown to the model with the judge's reading.
- `move_to` jumps to a new tilt with no ramp. Leaning in one large step makes
  the vial slide about 5 mm down the fingers each time until it reaches the
  pan. The API text says so, and the feedback reports the slide and whether
  the vial ended on the pan or on its side.
- No CaP-X binding for this task yet.

## Learning the weighing task in parts

`weigh_handover` and `weigh_pour` are the two halves of `weigh_salt`, with the
same layouts and targets seed for seed. In the first the judge wants the vial
standing in a cell of the middle rack, let go of, nothing spilled; in the
second the vial starts in the middle rack and the judge is the balance. The
first ends exactly where the second starts, so `--compose A/library B/library`
scores the two learned scripts run one after the other on the full task, with
no model call.

`--modules NAME=FOLDER ...` loads the libraries of earlier runs as fixed
modules: each is loaded on its own, the model calls it as `NAME.function(...)`
and cannot rewrite it, and the prompt shows its code and the script that used
it. `--inherit` still gives a starting library the model may overwrite.

```
RUN=a TASK=weigh_handover ... sbatch -n 12 cluster/run_evolve.sh
RUN=b TASK=weigh_pour ...     sbatch -n 12 cluster/run_evolve.sh
RUN=c TASK=weigh_salt EXTRA="... --modules handover=biolab_runs/a/library pour=biolab_runs/b/library" \
    sbatch -n 12 cluster/run_evolve.sh
```


## Tasks 4 and 5: protein assay and cell culture

Both use the right arm only, holding a pipette from the start. Liquids are
bookkeeping, not fluid simulation (`biolab/liquid.py`): a well holds a volume
and a concentration, liquid is always mixed, the tip must be in a well and
under the surface (or at the floor) to draw liquid, and dispensing outside a
well spills. `biolab/protein.py` holds the shared pipetting base.

`protein_assay` (`biolab/protein.py`): make six protein standards in a 2 x 4
plate from a stock and buffer, load an unknown sample, read the plate, report
the sample's concentration. The dye's response saturates and changes every
episode, so the unknown can only be measured against that episode's
standards. Judged on the standards (within 10 percent and 10 uL), the sample
well, and the report (within 10 percent). Reference: 52/52 layouts.

`cell_culture` (`biolab/culture.py`): keep a cancer cell line growing for a
simulated week in three dishes, with fresh medium, trypsin and waste
reservoirs and a 1 mL pipette. A small model grows the cells logistically to
about a million per dish while the medium lasts; they starve when it is
spent, die slowly when overgrown, lift off in trypsin and settle again once
it is diluted. Doubling time (20-28 h) and nutrient use vary and are hidden;
`inspect(dish)` gives noisy confluence, a cell estimate and the medium's
colour. Only `incubate(hours)` moves the culture clock. Judged at 168-180 h:
at least 1.8 million attached cells, 90 percent viable, no dish starved over
6 h or overgrown over 12 h, every dish with cells covered. Reference: 52/52.

```
RUN=x TASK=protein_assay MODEL=... EXTRA="--practice-seeds 100-111 --video --no-early-stop" sbatch -N 1 -n 12 cluster/run_evolve.sh
RUN=y TASK=cell_culture  MODEL=... EXTRA="--practice-seeds 100-111 --video --no-early-stop" sbatch -N 1 -n 12 cluster/run_evolve.sh
```

Reference videos (no model, no cost): `sbatch cluster/record_reference.sh` writes
`previews/<task>_reference.mp4` and three stills for `protein_assay` and `cell_culture`, layouts 0-2
(`TASKS=... SEEDS=...` to change). The overlay shows the pipette volume and the report for the assay,
and the day, clock and attached cells per dish for the culture; each incubation holds the frame for a
second so the clock jump can be read. Practice-round videos of these tasks get the same overlay.
Measured: 3 min, 1.8 GB, 3 cores for both tasks.

## Things to know

- `perceive()` returns ground-truth poses. It is a stand-in until camera-based
  perception is added; an agent using it is getting privileged information.
- The YAM can point its gripper straight down only within about 0.10-0.35 m of
  its base and below z = 0.16 m (wrist pitch limit). Above that, pass `tilt`.
  The racks sit inside this envelope and the vials are 75 mm tall for this reason.
- Both racks are in reach of the right arm only. The left arm is in the scene
  and controllable but has nothing to do in this task.
- The two vials never start in touching cells (diagonals included). The fingers
  need a free cell on each side, and with a third vial the finger struts grazed
  the diagonal neighbour in about a quarter of layouts.
- The gripper actuator is stiffer than Menagerie's (40 N max instead of about
  4 N), and `move_to` trims out the servo's friction offset at the end of each move.
- The wrist camera pose is a guess, not the real D405 mount.
