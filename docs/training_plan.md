# Stage-Wise Training Plan

## Stage 1: broad SFT

Purpose:

- activate broad reasoning capability already present in the base model
- cover math, open QA, instruction following, structured outputs, code, and
  self-correction

## Stage 2: hardening

Purpose:

- upweight borderline and hard examples
- focus on failure replay and stronger constraint following

## Stage 3: calibration

Purpose:

- reduce overlong reasoning
- improve final answer stability
- improve structured output validity

## Optional Stage 4: RLVR

Purpose:

- refine decision boundaries on verifier-rich subsets
- prioritize math, open QA, structured outputs, and code

## Evaluation slices

Track at least:

- overall normalized match
- hard bucket accuracy
- short reasoning accuracy
- answer-only accuracy
- invalid output rate
- overlong output rate

