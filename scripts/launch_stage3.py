#!/usr/bin/env python3
"""Prepare or launch a stage3 training bundle."""

from kaggle_nvida.training.launchers import run_stage_cli


if __name__ == "__main__":
    raise SystemExit(run_stage_cli("stage3"))

