# Project Architecture

## Goal

This repository turns the competition strategy into a practical workflow:

1. generate training candidates
2. normalize and score examples
3. filter and deduplicate them
4. build stage-specific mixtures
5. track LoRA experiments against consistent manifests

## Core modules

### `kaggle_nvida.schemas`

Defines the training example, quality metadata, provenance metadata, and
manifest record used across the repository.

### `kaggle_nvida.synthesis`

Contains candidate generators. The first version focuses on:

- arithmetic and algebra-style synthetic math
- candidate-selection examples
- self-correction examples

### `kaggle_nvida.curation`

Implements:

- normalization
- exact and fuzzy deduplication
- heuristic quality checks
- verifier hooks
- stage mixture construction

### `kaggle_nvida.tracking`

Creates experiment folders, writes experiment metadata, and records run
summaries for LoRA sweeps.

## Data flow

```text
raw candidates
  -> normalized jsonl
  -> filtered + scored manifest
  -> stage mixture manifests
  -> training exports
```

## Guiding principles

- every example is traceable back to a source and transform chain
- variants of the same problem share a common `problem_id`
- stage assignment is explicit in metadata
- filtering decisions are recorded rather than hidden

