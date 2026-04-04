# Evaluation Slices

## Goal

Break evaluation into reusable slices so every run can be inspected beyond a
single aggregate score.

## Included tools

- `build-eval-slices`: create slice manifests from a stage selection manifest
- `score-predictions`: compare model predictions against one slice manifest

## Default slices

- `overall_top`
- `math_hard`
- `retrieval_grounded`
- `structured_strict`
- `short_reasoning`
- `repair_and_selection`

## Example

```bash
PYTHONPATH=src python3 -m kaggle_nvida.cli build-eval-slices \
  --manifest outputs/stage1_selection.jsonl \
  --output-dir outputs/eval_slices
```

```bash
PYTHONPATH=src python3 -m kaggle_nvida.cli score-predictions \
  --slice-manifest outputs/eval_slices/overall_top.jsonl \
  --predictions outputs/predictions.jsonl \
  --output outputs/overall_top_metrics.json
```

