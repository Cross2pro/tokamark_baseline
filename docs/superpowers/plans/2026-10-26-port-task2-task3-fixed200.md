# Port Task 2/3 (fixed 200 shots, CNN vs TCN) Implementation Plan

> **CURRENT (2026-10-26):** Do **not** use batch_00000-only 200. Use the signal-complete pin:
> - Lists: `tokamark_baseline/src/config/fixed_shots_200.txt` + `fixed_split_200.csv` + `fixed_shots_200_sources.csv`
> - Data: `/home/remote/lee/datasets/MAST-zarrzip-fixed200/` (200 `.zarr.zip`)
> - Rules: `docs/subset-selection-rules.md` (subset must cover all Task 2/3/4 I/O signal keys)
> - Configs: `config_model_subset_{cnn,tcn}.yaml` → `output_subset_v2`, `base_local_zarr_path=MAST-zarrzip-fixed200`
> - Legacy incomplete list: `fixed_shots_200_batch00000_legacy.txt`


> **UPDATE 2026-10-26 (later):** Fixed-200 replaced. Old batch_00000 list is `fixed_shots_200_batch00000_legacy.txt`. New list is signal-complete for Task 2/3/4. Follow `docs/subset-selection-rules.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On two fresh GPU servers, reproduce the TokaMark CNN vs TCN comparison for Task 2-1/2-2/2-3 (server A) and Task 3-1/3-2/3-3 (server B) on the *exact same* pinned 200 MAST shots, then ship metrics back to the origin machine.

**Architecture:** Clone Cross2pro forks (`https://github.com/Cross2pro/tokamark` @ `feat/zarr-zip-store`, `https://github.com/Cross2pro/tokamark_baseline` @ `feat/tcn-backbone`), rebuild the Python 3.13 venv, copy the pinned 200 `<shot>.zarr.zip` files from `MAST-zarrzip-fixed200` (or extract them from the listed `batch_*.tar` via `fixed_shots_200_sources.csv`), verify checksums, rewrite only the absolute paths in the subset YAMLs, run a 20-step smoke per task/model, then run the sequential queue CNN→eval→TCN→eval per task and summarize `task_metrics.csv`.

**Tech Stack:** Python 3.13.5, torch 2.6.0+cu124 (CUDA 12.4, cuDNN 90100), zarr 3.1.5, xarray 2026.9.0, numpy 2.2.6, pandas 2.3.3, PyYAML 6.0.3; repos `tokamark` (branch `feat/zarr-zip-store`) and `tokamark_baseline` (branch `feat/tcn-backbone`); origin GPU: 1x RTX 3090, driver 570.190.

## Global Constraints

- **REQUIRED:** Read `docs/subset-selection-rules.md` first. Subset MUST cover all Task 2/3/4 input/actuator/output signal keys; shots missing any key are forbidden. Always run `scripts/scan_signal_coverage.py` before selecting a new subset.
- Data = the **signal-complete** pinned 200 shots (NOT the legacy batch_00000 list). Canonical files on origin:
  - `tokamark_baseline/src/config/fixed_shots_200.txt` — sha256 `a38dcc9366e07e4aa3c5b67e29324bf8a1be6e4764883f0a996491b443716f84`
  - `tokamark_baseline/src/config/fixed_split_200.csv` — sha256 `f84675315948e59731612345947a9991aa8df4444b7fb660c718e61b90f521e2` (train 172 / val 10 / test 18, seed 23)
  - `tokamark_baseline/src/config/fixed_shots_200_sources.csv` — sha256 `a9d6024d0dc421d3929883b91fd27e7bb509225d37159920dc0eb5f4a0130fb8` (shot → source tar; spans **32** tars)
  - Legacy (incomplete signals, do not use): `fixed_shots_200_batch00000_legacy.txt`
- Preferred data landing on each server: directory of 200 `<shot>.zarr.zip` files (SMR-friendly). Origin path: `/home/remote/lee/datasets/MAST-zarrzip-fixed200/`. Extract with `scripts/extract_fixed200_by_offset.py` from the corresponding `batch_*.tar` members listed in `fixed_shots_200_sources.csv`. Do **not** unpack zips into millions of small files on SMR disks.
- Split: NEVER regenerate / reshuffle. Use the pinned CSV only.
- Models = `cnn` and `tcn` only. Do NOT run `lstm`.
- Hyperparameters = `config_model_subset_{cnn,tcn}.yaml` (identical): `subset_of_shots:` empty, `shuffle: false`, `local: true`, `batch_size: 32`, `num_workers: 0`, `lr: 0.0001`, `max_steps: 5000`, `patience: 10`, `data_output_directory: ./output_subset_v2`. CLI: `--seed 23 --split random`. Only `data_splits_file` and `base_local_zarr_path` may be edited for new-machine paths. Point `base_local_zarr_path` at the 200-zip directory.
- `num_workers` stays `0`.
- `--config` must be repo-relative starting with `/src/config/...` (code does `REPO_ROOT + args.config`).
- `export PYTHONPATH=$WS/tokamark/src`; run from `$WS/tokamark_baseline`.
- Checkpoints: `$WS/tokamark_baseline./output_subset_v2/random/<model>/<task>/seed_23/` (trailing-dot quirk). Metrics: `$WS/tokamark_baseline/results/random/<model>/seed_23/<task>/task_metrics.csv`.
- Split of work: **server A = task_2-1..2-3**; **server B = task_3-1..3-3**. Origin host continues Task 4 on this same fixed-200. One GPU job at a time per server.
- Do not `git push` to UKAEA upstream; push feature work to Cross2pro forks only when the user asks.

---

## File Structure

| Path (new server) | Responsibility |
|---|---|
| `$WS/tokamark/` | `https://github.com/Cross2pro/tokamark` branch `feat/zarr-zip-store` (pin HEAD to `02bff918b06ac5b1eec0758093f4db949b1d2382` or later tip of that branch) |
| `$WS/tokamark_baseline/` | `https://github.com/Cross2pro/tokamark_baseline` branch `feat/tcn-backbone` (pin HEAD to `031c07ed5dd0c90bf79848e9c97c0b8ac8945125` or later tip) |
| `$WS/tokamark-venv313/` | Python 3.13 venv |
| `$WS/transfer/` | `batch_00000.tar` (or empty if Option B used) + checksum logs |
| `$WS/run_port_queue.sh` | Create: sequential CNN→eval→TCN→eval queue for this server's tasks |
| `$WS/summarize_port.py` | Create: builds comparison table from `task_metrics.csv` |
| `$WS/logs/` | per-run logs + `port_status.tsv` |

Facts verified on origin (2026-10-26):
- `run_training.py:143-147` known-task set includes task_2-1/2-2/2-3 and task_3-1/3-2/3-3. `run_training.py:132-138`: task_3-3 (and 4-x) use `stride_window=0.025`; every other task, including all Task 2 and 3-1/3-2, uses `stride_window=0.005` (overrides the YAML's 0.001). `run_evaluation.py:106` accepts `--model cnn|lstm|tcn`.
- Task YAMLs: `tokamark/src/tokamark/tasks_configs/group_2_magnetics_dynamics/task_2-{1,2,3}.yaml`, `group_3_profiles_dynamics/task_3-{1,2,3}.yaml`.
- Task 2 (all, markovian, in 5 ms → out 25 ms): inputs magnetics{flux_loop_flux, b_field_pol_probe_ccbv_field, _obr_field, _obv_field, b_field_tor_probe_saddle_voltage}, pf_active{coil_current, solenoid_current}, summary.ip; actuators pf_active.coil_voltage, summary.power_nbi. Outputs: 2-1 = coil/solenoid current, ip + 15 equilibrium scalars; 2-2 = equilibrium.lcfs_r/lcfs_z; 2-3 = equilibrium.psi.
- Task 3-1 (in 5 ms → out 50 ms): thomson_scattering.t_e/n_e → same; actuators pulse_schedule.i_plasma/n_e_line, summary.power_nbi, gas_injection.total_injected. Task 3-2: adds spectrometer_visible.filter_spectrometer_dalpha_voltage, soft_x_rays.horizontal_cam_lower/upper. Task 3-3 (non_markovian, in 150 ms → out 5 ms): Task 2 magnetics inputs + interferometer.n_e_line → thomson t_e/n_e + beta_tor/pol/normal.
- **Signal-complete subset (2026-10-26):** selected from 4247 qualified shots (full Task 2+3+4 key coverage scan over 11573 shots; qualify rate ~36.7%). Pinned 200 across 32 tars, seed=23. Legacy batch_00000 list is incomplete (missing `gas_injection/total_injected` etc.) — do not use. See `docs/subset-selection-rules.md` and `logs/signal_scan/`.

---

### Task 1: Environment (code + venv)

**Files:**
- Create: `$WS/tokamark/`, `$WS/tokamark_baseline/`, `$WS/tokamark-venv313/`, `$WS/requirements-lock-venv313.txt`

**Interfaces:**
- Consumes: GitHub forks `https://github.com/Cross2pro/tokamark` (branch `feat/zarr-zip-store`), `https://github.com/Cross2pro/tokamark_baseline` (branch `feat/tcn-backbone`); origin lock file `$ORIGIN:/home/remote/lee/gh-pr/tokamark_workspace/requirements-lock-venv313.txt`
- Produces: `$WS/tokamark-venv313/bin/python` importing `torch`, `tokamark`; repos on the named branches (origin tip was `02bff918…` / `031c07ed…`).

- [ ] **Step 1: Create workspace and copy lock file**

```bash
export WS=/data/tokamark_workspace   # choose; keep for all tasks
mkdir -p $WS/transfer $WS/logs && cd $WS
scp "$ORIGIN:/home/remote/lee/gh-pr/tokamark_workspace/requirements-lock-venv313.txt" .
wc -l requirements-lock-venv313.txt
```
Expected: `225 requirements-lock-venv313.txt` (or nearby; used only as a hint — Step 3 filters conda `file://` lines).

- [ ] **Step 2: Clone Cross2pro forks and checkout feature branches**

Do **not** clone `UKAEA-IBM-STFC-Fusion-FMs/*` for the feature work. Use Cross2pro forks:

```bash
cd $WS
git clone https://github.com/Cross2pro/tokamark.git
git clone https://github.com/Cross2pro/tokamark_baseline.git
(cd tokamark && git fetch origin && git checkout feat/zarr-zip-store && git log -1 --format='%H %s')
(cd tokamark_baseline && git fetch origin && git checkout feat/tcn-backbone && git log -1 --format='%H %s')
```
Expected: branch names above; HEADs contain at least the origin commits (tokamark `02bff918b06ac5b1eec0758093f4db949b1d2382` "feat: read <shot>.zarr.zip…"; baseline `031c07ed5dd0c90bf79848e9c97c0b8ac8945125` "docs: Task 4-x only on this host…"). If clone 404s, wait ~1–2 min and retry (fork may still be propagating). If the branch exists but tip differs, use tip of the branch (do not force-reset unless the user asks).

Optional fallback (only if GitHub is unreachable): origin also has git bundles at `$ORIGIN:.../transfer/tokamark_feat-zarr-zip-store.bundle` (sha256 `3d128501e2251824d182cae8c5ffbb4a21da500e390008eec3127fb1f9e35175`) and `tokamark_baseline_feat-tcn-backbone.bundle` (sha256 `9752d76279a86909f60a296b0e78df3b7889f7680dcf8f29d0e58e9d3e04483e`); `git clone` UKAEA main first, then `git fetch <bundle> <branch>:<branch>` — prefer the forks.

- [ ] **Step 3: Build venv**

Origin venv is `python3.13 -m venv --system-site-packages` on top of miniconda; its `pip freeze` contains conda `file:///croot/...` lines that cannot be installed elsewhere. Install only the pinned research stack:

```bash
cd $WS
python3.13 --version            # need 3.13.x (install via conda/uv if missing: `conda create -p $WS/py313 python=3.13.5`)
python3.13 -m venv tokamark-venv313
source tokamark-venv313/bin/activate
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
grep -vE '^(torch|torchvision)==|@ file://|^-e |^#' requirements-lock-venv313.txt > req-portable.txt
pip install -r req-portable.txt || pip install numpy==2.2.6 pandas==2.3.3 PyYAML==6.0.3 xarray==2026.9.0 zarr==3.1.5 torchinfo==1.8.0 tqdm
pip install -e $WS/tokamark
```
If `req-portable.txt` fails on some unrelated package, use the fallback list (already in the `||`).

- [ ] **Step 4: Verify**

```bash
$WS/tokamark-venv313/bin/python -c "import sys,torch,zarr,tokamark;print(sys.version.split()[0],torch.__version__,torch.version.cuda,torch.cuda.is_available(),zarr.__version__)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
(cd $WS/tokamark && git remote -v | head -1)
(cd $WS/tokamark_baseline && git remote -v | head -1)
```
Expected: `3.13.x 2.6.0+cu124 12.4 True 3.1.5`; a GPU line with driver ≥ 550; remotes pointing at `github.com/Cross2pro/tokamark` and `…/tokamark_baseline`. Success = CUDA `True`.

---

### Task 2: Data transfer and verification

**Files:**
- Create: `$DATA/<shot>.zarr` ×200, `$WS/transfer/batch_00000.tar` (option B only)

**Interfaces:**
- Consumes: origin `/home/remote/lee/datasets/MAST-zarr/` (200 dirs) or `/home/remote/lee/datasets/MAST-dataset/batch_00000.tar`
- Produces: `$DATA` containing exactly the 200 shots in `fixed_shots_200.txt`.

- [ ] **Step 1: Copy data (pick one)**

Option A — copy the tar (1 file, 1.94 GB, fastest over network; origin disk is a slow SMR HDD so avoid millions of small-file reads there):
```bash
export DATA=/data/MAST-zarr
rsync -av --progress "$ORIGIN:/home/remote/lee/datasets/MAST-dataset/batch_00000.tar" $WS/transfer/
stat -c %s $WS/transfer/batch_00000.tar
sha256sum $WS/transfer/batch_00000.tar | tee $WS/transfer/batch_00000.tar.sha256
```
Expected: size `1941278720`. Store the sha256 in the transfer log (origin was still hashing the SMR disk when this plan was written; the new-machine hash is authoritative after a clean copy). Size mismatch = stop.

Then extract: the tar holds 200 `<shot>.zarr.zip`; unzip each to a `<shot>.zarr` dir (the subset configs read extracted dirs):
```bash
mkdir -p $DATA $WS/transfer/b0 && tar -xf $WS/transfer/batch_00000.tar -C $WS/transfer/b0
find $WS/transfer/b0 -name '*.zarr.zip' | wc -l       # expect 200
for z in $(find $WS/transfer/b0 -name '*.zarr.zip'); do s=$(basename $z .zarr.zip); mkdir -p $DATA/$s.zarr && unzip -q -o $z -d $DATA/$s.zarr; done
```
Check that `$DATA/11940.zarr/magnetics` exists (not `$DATA/11940.zarr/11940.zarr/magnetics`). If there is an extra nesting level, `mv` the inner dir up.

Option B — copy already-extracted dirs:
```bash
rsync -a "$ORIGIN:/home/remote/lee/datasets/MAST-zarr/" $DATA/
```

- [ ] **Step 2: Verify shot list**

```bash
cd $WS/tokamark_baseline/src/config
sha256sum fixed_shots_200.txt fixed_split_200.csv
diff <(sort fixed_shots_200.txt) <(ls $DATA | sed 's/\.zarr$//' | sort) && echo SHOTS_MATCH
python3 -c "import csv;r=list(csv.DictReader(open('fixed_split_200.csv')));print({k:sum(x[k]=='True' for x in r) for k in ['train','val','test']})"
```
Expected:
```
ba11c39524306d4b7e727bb105c695e55118033829ac1b17ebbbcc211d24b59f  fixed_shots_200.txt
d949ad15a82e2252a7b3dd215603f0d22a7dea41c127d2371024a125633918d8  fixed_split_200.csv
SHOTS_MATCH
{'train': 172, 'val': 10, 'test': 18}
```

- [ ] **Step 3: Verify signal groups**

```bash
for s in 11766 11940 12093; do echo "## $s"; for g in magnetics pf_active summary equilibrium thomson_scattering pulse_schedule gas_injection spectrometer_visible soft_x_rays interferometer; do echo "$g: $(ls $DATA/$s.zarr/$g 2>/dev/null | tr '\n' ' ')"; done; done
```
Expected (matches origin): 11940 and 12093 list `flux_loop_flux`, `coil_voltage`, `power_nbi`, `psi`, `lcfs_r`, `t_e`, `n_e`, `horizontal_cam_lower`, `n_e_line`; 11766 has empty `thomson_scattering:` and `soft_x_rays:`; `gas_injection:` shows `inboard_total outboard_total ...` with no `total_injected` in all three.

---

### Task 3: Point configs at new paths

**Files:**
- Modify: `$WS/tokamark_baseline/src/config/config_model_subset_cnn.yaml` (lines `data_splits_file:` and `base_local_zarr_path:`)
- Modify: `$WS/tokamark_baseline/src/config/config_model_subset_tcn.yaml` (same lines)
- Create: `$WS/tokamark_baseline/src/config/config_model_smoke20_{cnn,tcn}.yaml`

**Interfaces:**
- Consumes: `$DATA`, `fixed_split_200.csv`
- Produces: configs `/src/config/config_model_subset_{cnn,tcn}.yaml` and `/src/config/config_model_smoke20_{cnn,tcn}.yaml` (passed via `--config`).

- [ ] **Step 1: Rewrite the two absolute paths**

On origin the YAML points to `/home/remote/lee/gh-pr/tokamark_workspace/fixed_split_200.csv` (byte-identical to the in-repo copy) and `/home/remote/lee/datasets/MAST-zarr`.
```bash
cd $WS/tokamark_baseline/src/config
for m in cnn tcn; do
  sed -i "s#^data_splits_file:.*#data_splits_file: $WS/tokamark_baseline/src/config/fixed_split_200.csv#; s#^\(  base_local_zarr_path:\).*#\1 $DATA#" config_model_subset_$m.yaml
done
diff config_model_subset_cnn.yaml config_model_subset_tcn.yaml && echo IDENTICAL
grep -nE 'data_splits_file|base_local_zarr_path|batch_size|num_workers|lr:|max_steps|patience|data_output_directory|shuffle|local' config_model_subset_cnn.yaml
```
Expected: `IDENTICAL`, new paths, and `batch_size: 32`, `num_workers: 0`, `lr: 0.0001`, `max_steps: 5000`, `patience: 10`, `data_output_directory: ./output_subset`, `shuffle: false`, `local: true`.

- [ ] **Step 2: Create smoke configs (20 steps, separate output dir)**

```bash
for m in cnn tcn; do sed 's/^  max_steps:.*/  max_steps: 20/; s#^  data_output_directory:.*#  data_output_directory: ./output_smoke20#' config_model_subset_$m.yaml > config_model_smoke20_$m.yaml; done
grep -E 'max_steps|data_output_directory' config_model_smoke20_cnn.yaml
```
Expected: `max_steps: 20` and `data_output_directory: ./output_smoke20`.

- [ ] **Step 3: Commit locally (no push)**

```bash
cd $WS/tokamark_baseline && git add src/config/config_model_subset_*.yaml src/config/config_model_smoke20_*.yaml && git commit -m "chore: point fixed-200 configs at $(hostname) paths"
```

---

### Task 4: Smoke test (20 steps per task × model)

**Files:**
- Create: `$WS/logs/smoke20_<task>_<model>.log`

**Interfaces:**
- Consumes: configs from Task 3
- Produces: confirmation that each of this server's tasks trains and evaluates.

- [ ] **Step 1: Run smoke** (server A: `TASKS="task_2-1 task_2-2 task_2-3"`; server B: `TASKS="task_3-1 task_3-2 task_3-3"`)

```bash
cd $WS/tokamark_baseline && export PYTHONPATH=$WS/tokamark/src
PY=$WS/tokamark-venv313/bin/python
for t in $TASKS; do for m in cnn tcn; do
  L=$WS/logs/smoke20_${t}_${m}.log
  $PY -u run_training.py --task $t --config /src/config/config_model_smoke20_$m.yaml --seed 23 --model $m --split random > $L 2>&1 && \
  $PY -u run_evaluation.py --task $t --config /src/config/config_model_smoke20_$m.yaml --seed 23 --model $m --split random >> $L 2>&1
  echo "$t $m rc=$?"
done; done
```
Expected: `rc=0` for all 6 lines; each log contains `[INFO] task <task>: num_workers=0`; `ls $WS/tokamark_baseline./output_smoke20/random/<model>/<task>/seed_23/best_model.pt` exists.
Note: eval writes to `results/random/<model>/seed_23/<task>/` too; delete smoke metrics before the real run: `rm -rf $WS/tokamark_baseline/results/random/*/seed_23/task_[23]-*`.

- [ ] **Step 2: Triage failures**

If a run fails: `tail -50` its log. CUDA OOM on task_2-3 (2D psi maps) → do NOT change batch_size silently; report to the user with the log tail (changing it breaks comparability). Empty dataset / zero windows → recheck Task 2 Step 2–3. Anything else → use superpowers:systematic-debugging, report before changing code.

---

### Task 5 (server A): Task 2-1, 2-2, 2-3 formal runs

**Files:**
- Create: `$WS/run_port_queue.sh`
- Create: `$WS/logs/port_<task>_<model>{,_eval}.log`, `$WS/logs/port_status.tsv`

**Interfaces:**
- Consumes: `/src/config/config_model_subset_{cnn,tcn}.yaml`
- Produces: `$WS/tokamark_baseline/results/random/{cnn,tcn}/seed_23/task_2-{1,2,3}/task_metrics.csv`, checkpoints in `$WS/tokamark_baseline./output_subset/random/{cnn,tcn}/task_2-*/seed_23/best_model.pt`

- [ ] **Step 1: Write queue script** (identical on both servers; tasks passed as args)

```bash
cat > $WS/run_port_queue.sh <<'SH'
#!/usr/bin/env bash
# Sequential CNN->eval->TCN->eval on pinned 200-shot subset. No LSTM.
set -u
WS=${WS:?set WS}
BL=$WS/tokamark_baseline
PY=$WS/tokamark-venv313/bin/python
export PYTHONPATH=$WS/tokamark/src
cd "$BL"; mkdir -p "$WS/logs"
STATUS=$WS/logs/port_status.tsv
[[ -f $STATUS ]] || echo -e "task\tmodel\tphase\trc\ttime" > "$STATUS"
ts() { date +"%Y-%m-%dT%H:%M:%S%z"; }
for task in "$@"; do for model in cnn tcn; do
  cfg="/src/config/config_model_subset_${model}.yaml"
  tlog=$WS/logs/port_${task}_${model}.log; elog=$WS/logs/port_${task}_${model}_eval.log
  if grep -q "^${task}	${model}	eval	0" "$STATUS"; then echo "skip $task $model"; continue; fi
  echo "$(ts) START $task $model"
  "$PY" -u run_training.py --task "$task" --config "$cfg" --seed 23 --model "$model" --split random >"$tlog" 2>&1
  trc=$?; echo -e "${task}\t${model}\ttrain\t${trc}\t$(ts)" >> "$STATUS"
  [[ $trc -ne 0 ]] && continue
  "$PY" -u run_evaluation.py --task "$task" --config "$cfg" --seed 23 --model "$model" --split random >"$elog" 2>&1
  echo -e "${task}\t${model}\teval\t$?\t$(ts)" >> "$STATUS"
done; done
echo "$(ts) QUEUE_DONE"
SH
chmod +x $WS/run_port_queue.sh && bash -n $WS/run_port_queue.sh && echo SYNTAX_OK
```
Expected: `SYNTAX_OK`.

- [ ] **Step 2: Launch (detached)**

```bash
cd $WS && export WS && nohup ./run_port_queue.sh task_2-1 task_2-2 task_2-3 > logs/port_queue.out 2>&1 &
sleep 60; tail -3 logs/port_queue.out; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```
Expected: `START task_2-1 cnn`, GPU memory used > 1 GB. Origin reference: one model on 200 shots took ~20–40 min on an RTX 3090 (Task 4-1: CNN ~22 min, TCN ~34 min, early-stopped).

- [ ] **Step 3: Verify completion**

```bash
tail -1 $WS/logs/port_queue.out; column -t $WS/logs/port_status.tsv
ls $WS/tokamark_baseline/results/random/{cnn,tcn}/seed_23/task_2-{1,2,3}/task_metrics.csv
```
Expected: `QUEUE_DONE`; 12 status rows all `rc` 0 (train+eval × 3 tasks × 2 models); 6 csv paths listed. Each CSV has header `feature_name,n_shots,NRMSE_mean,NRMSE_std_pop,NMAE_mean,NMAE_std_pop,RMSE_mean,RMSE_std_pop,MAE_mean,MAE_std_pop,nan_fraction_mean,nan_fraction_std_pop` and a row whose `feature_name` equals the task name with `n_shots` ≤ 18. A re-run of the script skips finished pairs.

---

### Task 6 (server B): Task 3-1, 3-2, 3-3 formal runs

**Files:**
- Create: `$WS/run_port_queue.sh`
- Create: `$WS/logs/port_<task>_<model>{,_eval}.log`, `$WS/logs/port_status.tsv`

**Interfaces:**
- Consumes: `/src/config/config_model_subset_{cnn,tcn}.yaml`
- Produces: `$WS/tokamark_baseline/results/random/{cnn,tcn}/seed_23/task_3-{1,2,3}/task_metrics.csv`

- [ ] **Step 1: Write queue script**

```bash
cat > $WS/run_port_queue.sh <<'SH'
#!/usr/bin/env bash
# Sequential CNN->eval->TCN->eval on pinned 200-shot subset. No LSTM.
set -u
WS=${WS:?set WS}
BL=$WS/tokamark_baseline
PY=$WS/tokamark-venv313/bin/python
export PYTHONPATH=$WS/tokamark/src
cd "$BL"; mkdir -p "$WS/logs"
STATUS=$WS/logs/port_status.tsv
[[ -f $STATUS ]] || echo -e "task\tmodel\tphase\trc\ttime" > "$STATUS"
ts() { date +"%Y-%m-%dT%H:%M:%S%z"; }
for task in "$@"; do for model in cnn tcn; do
  cfg="/src/config/config_model_subset_${model}.yaml"
  tlog=$WS/logs/port_${task}_${model}.log; elog=$WS/logs/port_${task}_${model}_eval.log
  if grep -q "^${task}	${model}	eval	0" "$STATUS"; then echo "skip $task $model"; continue; fi
  echo "$(ts) START $task $model"
  "$PY" -u run_training.py --task "$task" --config "$cfg" --seed 23 --model "$model" --split random >"$tlog" 2>&1
  trc=$?; echo -e "${task}\t${model}\ttrain\t${trc}\t$(ts)" >> "$STATUS"
  [[ $trc -ne 0 ]] && continue
  "$PY" -u run_evaluation.py --task "$task" --config "$cfg" --seed 23 --model "$model" --split random >"$elog" 2>&1
  echo -e "${task}\t${model}\teval\t$?\t$(ts)" >> "$STATUS"
done; done
echo "$(ts) QUEUE_DONE"
SH
chmod +x $WS/run_port_queue.sh && bash -n $WS/run_port_queue.sh && echo SYNTAX_OK
```
Expected: `SYNTAX_OK`.

- [ ] **Step 2: Launch**

```bash
cd $WS && export WS && nohup ./run_port_queue.sh task_3-1 task_3-2 task_3-3 > logs/port_queue.out 2>&1 &
sleep 60; tail -3 logs/port_queue.out
```
Expected: `START task_3-1 cnn`. Task 3-3 uses stride 0.025 and a 150 ms input window (fewer, longer windows); 3-1/3-2 use stride 0.005.

- [ ] **Step 3: Verify completion**

```bash
tail -1 $WS/logs/port_queue.out; column -t $WS/logs/port_status.tsv
ls $WS/tokamark_baseline/results/random/{cnn,tcn}/seed_23/task_3-{1,2,3}/task_metrics.csv
grep -h nan_fraction -A0 $WS/tokamark_baseline/results/random/cnn/seed_23/task_3-1/task_metrics.csv
```
Expected: `QUEUE_DONE`, 12 rows rc 0, 6 CSVs. Shots without Thomson data (e.g. 11766) are dropped by NaN filtering, so `n_shots` for Task 3 can be < 18 — record it, it is not an error.

---

### Task 7: Summary table

**Files:**
- Create: `$WS/summarize_port.py`, `$WS/results_port_<server>.md`

**Interfaces:**
- Consumes: `results/random/{cnn,tcn}/seed_23/<task>/task_metrics.csv`
- Produces: markdown table of the task-level row (`feature_name == task`) per model.

- [ ] **Step 1: Write script**

```python
# $WS/summarize_port.py
import sys, os, pandas as pd
ws = os.environ["WS"]; tasks = sys.argv[1:]
rows = []
for t in tasks:
    for m in ("cnn", "tcn"):
        p = f"{ws}/tokamark_baseline/results/random/{m}/seed_23/{t}/task_metrics.csv"
        if not os.path.exists(p):
            rows.append({"task": t, "model": m, "status": "MISSING"}); continue
        d = pd.read_csv(p); r = d[d.feature_name == t].iloc[0]
        rows.append({"task": t, "model": m, "n_shots": int(r.n_shots), "NRMSE": r.NRMSE_mean, "NMAE": r.NMAE_mean, "RMSE": r.RMSE_mean, "MAE": r.MAE_mean, "status": "ok"})
print(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))
```

- [ ] **Step 2: Run**

```bash
cd $WS && pip install tabulate >/dev/null; WS=$WS tokamark-venv313/bin/python summarize_port.py $TASKS | tee results_port_$(hostname).md
```
Expected: 6 rows, all `status ok`, numeric NRMSE/NMAE/RMSE/MAE (lower is better).

---

### Task 8: Ship results back

**Files:**
- Create: `$WS/port_results_<server>.tar.gz`; on origin `/home/remote/lee/gh-pr/tokamark_workspace/ported_results/<hostname>/`

**Interfaces:**
- Consumes: outputs of Tasks 5–7
- Produces: origin copy of metrics, logs, checkpoints, env info.

- [ ] **Step 1: Pack**

```bash
cd $WS
{ hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; tokamark-venv313/bin/python -c "import torch;print(torch.__version__)"; (cd tokamark && git rev-parse HEAD); (cd tokamark_baseline && git rev-parse HEAD); } > env_info.txt
tokamark-venv313/bin/pip freeze > pip_freeze_$(hostname).txt
tar -czf port_results_$(hostname).tar.gz env_info.txt pip_freeze_$(hostname).txt results_port_$(hostname).md logs/port_* \
  tokamark_baseline/results/random/{cnn,tcn}/seed_23/ "tokamark_baseline./output_subset/random/"
tar -tzf port_results_$(hostname).tar.gz | grep -c task_metrics.csv
```
Expected: count ≥ 6.

- [ ] **Step 2: Copy to origin**

```bash
ssh $ORIGIN "mkdir -p /home/remote/lee/gh-pr/tokamark_workspace/ported_results/$(hostname)"
scp port_results_$(hostname).tar.gz "$ORIGIN:/home/remote/lee/gh-pr/tokamark_workspace/ported_results/$(hostname)/"
sha256sum port_results_$(hostname).tar.gz; ssh $ORIGIN "sha256sum /home/remote/lee/gh-pr/tokamark_workspace/ported_results/$(hostname)/port_results_$(hostname).tar.gz"
```
Expected: both sha256 values identical. Report the markdown table and the note "gas_injection/total_injected absent in data (Task 3 actuator all-NaN)" to the user.

---

## Self-Review (done by plan author)
- Coverage: env via Cross2pro forks (T1), data+checksums (T2), path config (T3), 20-step smoke (T4), Task 2 (T5, server A), Task 3 (T6, server B), summary (T7), return (T8). No LSTM anywhere; split never regenerated.
- Placeholder scan: no TBD/TODO; Task 6 repeats the full queue script verbatim.
- Names consistent: `run_port_queue.sh`, `port_status.tsv`, `config_model_smoke20_*`, `summarize_port.py`, `$WS`/`$DATA`/`$ORIGIN`/`$TASKS`.
