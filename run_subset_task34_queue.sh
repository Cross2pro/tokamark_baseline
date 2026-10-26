#!/usr/bin/env bash
# Sequential CNN→eval→TCN→eval on pinned 200-shot subset (Task 4-x then 3-x).
# Skip task_4-1 (already done). Do not run LSTM.
set -u
WS=/home/remote/lee/gh-pr/tokamark_workspace
BL=$WS/tokamark_baseline
PY=$WS/tokamark-venv313/bin/python
export PYTHONPATH=$WS/tokamark/src
cd "$BL"
mkdir -p "$WS/logs"
QUEUE_LOG=$WS/logs/subset_task34_queue.log
STATUS=$WS/logs/subset_task34_status.tsv
echo -e "task\tmodel\tphase\trc\tnote\ttime" > "$STATUS"
ts() { date +"%Y-%m-%dT%H:%M:%S%z"; }

run_one() {
  local task=$1 model=$2
  local cfg="/src/config/config_model_subset_${model}.yaml"
  local tlog=$WS/logs/subset_${task}_${model}.log
  local elog=$WS/logs/subset_${task}_${model}_eval.log
  echo "$(ts) START $task $model train" | tee -a "$QUEUE_LOG"
  "$PY" -u run_training.py --task "$task" --config "$cfg" --seed 23 --model "$model" --split random \
    >"$tlog" 2>&1
  local trc=$?
  echo "$(ts) END $task $model train_rc=$trc" | tee -a "$QUEUE_LOG"
  if [[ $trc -ne 0 ]]; then
    # capture last error lines
    local note
    note=$(tail -30 "$tlog" | tr "\t\n" "  " | cut -c1-200)
    echo -e "${task}\t${model}\ttrain\t${trc}\t${note}\t$(ts)" >> "$STATUS"
    return 0  # continue queue
  fi
  echo -e "${task}\t${model}\ttrain\t0\tok\t$(ts)" >> "$STATUS"
  echo "$(ts) START $task $model eval" | tee -a "$QUEUE_LOG"
  "$PY" -u run_evaluation.py --task "$task" --config "$cfg" --seed 23 --model "$model" --split random \
    >"$elog" 2>&1
  local erc=$?
  echo "$(ts) END $task $model eval_rc=$erc" | tee -a "$QUEUE_LOG"
  if [[ $erc -ne 0 ]]; then
    local note
    note=$(tail -30 "$elog" | tr "\t\n" "  " | cut -c1-200)
    echo -e "${task}\t${model}\teval\t${erc}\t${note}\t$(ts)" >> "$STATUS"
  else
    echo -e "${task}\t${model}\teval\t0\tok\t$(ts)" >> "$STATUS"
  fi
}

# Task 4-1 already delivered — record reuse
echo -e "task_4-1\tcnn\treuse\t0\tprior subset result\t$(ts)" >> "$STATUS"
echo -e "task_4-1\ttcn\treuse\t0\tprior subset result\t$(ts)" >> "$STATUS"
echo "$(ts) SKIP task_4-1 (reuse existing checkpoints/metrics)" | tee -a "$QUEUE_LOG"

TASKS=(task_4-2 task_4-3 task_4-4 task_4-5 task_3-1 task_3-2 task_3-3)
for task in "${TASKS[@]}"; do
  for model in cnn tcn; do
    run_one "$task" "$model"
  done
done
echo "$(ts) QUEUE_DONE" | tee -a "$QUEUE_LOG"
