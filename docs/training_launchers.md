# Training Launchers

## Goal

Prepare reproducible run directories for `stage1`, `stage2`, and `stage3`
without hard-coding training commands into notebooks.

## Included pieces

- stage configs in `configs/training/`
- wrapper scripts in `scripts/launch_stage1.py`, `scripts/launch_stage2.py`,
  and `scripts/launch_stage3.py`
- a shared launcher module in `src/kaggle_nvida/training/launchers.py`

## Example

```bash
PYTHONPATH=src python3 scripts/launch_stage1.py \
  --manifest outputs/stage1_selection.jsonl \
  --run-dir runs/exp_stage1_a
```

This prepares:

- exported training data
- stage config snapshot
- tool stack config bundle
- a `launch_plan.json` with the intended external NeMo RL command

## Execute mode

If your environment already has a compatible `nemo_rl` launcher, add
`--execute` to run the external command after materialization.

