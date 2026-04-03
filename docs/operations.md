# Operations Guide

## Recommended order

1. generate or import candidate examples
2. build a manifest
3. score and curate the manifest
4. build a stage mixture
5. initialize a LoRA experiment run

## Minimal bootstrap flow

```bash
PYTHONPATH=src python3 -m kaggle_nvida.cli bootstrap-math \
  --output-dir outputs/bootstrap \
  --count 100

PYTHONPATH=src python3 -m kaggle_nvida.cli curate-jsonl \
  --dataset outputs/bootstrap/train.jsonl \
  --manifest outputs/bootstrap/manifest.jsonl \
  --output outputs/bootstrap/curated_manifest.jsonl

PYTHONPATH=src python3 -m kaggle_nvida.cli build-mixture \
  --manifest outputs/bootstrap/curated_manifest.jsonl \
  --stage stage1 \
  --output outputs/bootstrap/stage1_selection.jsonl \
  --max-examples 200

PYTHONPATH=src python3 -m kaggle_nvida.cli init-run \
  --runs-dir runs \
  --experiment-id exp_stage1_rank32_lr1e4
```

## What the first implementation covers

- deterministic synthetic bootstrap generation
- manifest creation with provenance and stage flags
- heuristic curation and dedup scoring
- stage-specific selection with target family ratios
- experiment directory and tracker initialization

## What still needs model-side integration

- tokenizer-aware token counting
- semantic dedup using embeddings
- stronger verifier and judge models
- integration with actual LoRA training scripts

