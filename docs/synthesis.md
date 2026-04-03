# Synthetic Data Guide

## Goal

Move beyond arithmetic-only bootstrap data and create a broader starter pack
that better matches hidden reasoning benchmarks.

## Current synthetic families

- `math`: arithmetic with full, brief, and answer-only variants
- `retrieval`: small document-grounded QA with explicit evidence
- `structured`: instruction-following tasks with strict JSON outputs

## Mixed bootstrap command

```bash
PYTHONPATH=src python3 -m kaggle_nvida.cli bootstrap-mixed \
  --output-dir outputs/mixed_bootstrap \
  --count-per-family 50
```

This produces one dataset JSONL and one manifest JSONL that can immediately be
sent into the curation stage.

## Design choices

- retrieval examples are fully synthetic, so the answer key is known
- structured examples emphasize exact fields and valid JSON outputs
- math still includes repair-oriented variants because those are useful in
  stage 2 and stage 3 mixtures

