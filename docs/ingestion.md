# Data Ingestion Guide

## Goal

Normalize external datasets into the repository training schema without tying
the pipeline to a single source format.

## Supported pattern

The first importer targets line-delimited JSON files and a small mapping spec.
This covers most public instruction, QA, and reasoning datasets once they are
converted into JSONL.

## Mapping file

The mapping file is a JSON document that tells the importer where to find:

- prompt or user text
- optional system prompt
- answer text
- optional reasoning text
- split and ids
- default task metadata
- which reasoning variants to generate

## Example flow

```bash
PYTHONPATH=src python3 -m kaggle_nvida.cli import-jsonl \
  --input examples/importer/raw_instruction.jsonl \
  --mapping examples/importer/mapping_instruction.json \
  --output outputs/imported/train.jsonl \
  --manifest-output outputs/imported/manifest.jsonl
```

## Notes

- imported rows are normalized into the same `TrainingExample` contract used by
  synthetic generators
- when a source only has answer text, the importer can still generate an
  `answer_only` variant
- when reasoning exists, the importer can emit `full_reasoning`,
  `brief_reasoning`, and `answer_only`

