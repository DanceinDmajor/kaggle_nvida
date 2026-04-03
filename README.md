# kaggle_nvida

Engineering workspace for the NVIDIA Nemotron Model Reasoning Challenge.

This repository is organized around a practical pipeline for:

- synthetic data generation
- data normalization and curation
- stage-wise manifest building
- LoRA experiment management

The first implementation target is a reproducible Python toolkit that can move
from candidate data generation to stage-specific training manifests with clear
provenance.

## Repository layout

```text
configs/      Training and mixture configuration templates
docs/         Longer design notes and execution plans
examples/     Small runnable example data and manifests
src/          Python source code
tests/        Unit tests for core utilities
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m kaggle_nvida.cli --help
```

## Current scope

- document the training-ready dataset contract
- implement synthetic reasoning data generators
- implement filtering, scoring, and stage mixture builders
- provide experiment tracking templates for LoRA runs
