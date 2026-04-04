# Tool Stack Bundles

## Goal

Create stage-specific configuration bundles that map the local project layout
onto `DataDesigner`, `Curator`, and `NeMo RL` style workflows.

## Why this layer exists

The repository does not depend on the external NVIDIA stacks directly, but it
needs a stable handoff point so a real environment can swap in:

- richer data generation recipes
- stronger curation pipelines
- NeMo RL launchers and trainers

## CLI

```bash
PYTHONPATH=src python3 -m kaggle_nvida.cli materialize-tool-stack \
  --stage stage1 \
  --manifest outputs/bootstrap/stage1_selection.jsonl \
  --train-export outputs/bootstrap/stage1_train.jsonl \
  --eval-slice-dir outputs/bootstrap/eval_slices \
  --output-dir outputs/bootstrap/tool_stack
```

## Files written

- `datadesigner_recipe.json`
- `curator_pipeline.json`
- `nemo_rl_recipe.json`
- `tool_stack_bundle.json`

## Stage behavior

### `stage1`

Bias toward broad coverage and higher-volume synthetic expansion.

### `stage2`

Bias toward hard examples, failure replay, and stronger curation.

### `stage3`

Bias toward calibration, structured output stability, and shorter reasoning.

