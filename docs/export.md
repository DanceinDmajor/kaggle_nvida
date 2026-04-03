# Training Export Guide

## Goal

Turn selected manifest rows into model-ready training files without manually
copying examples out of the curation layer.

## Supported output formats

### `chat_jsonl`

Writes normalized examples with the original message structure.

### `prompt_completion`

Writes one row per example with:

- `prompt`
- `completion`
- `example_id`
- `task_family`

### `tagged_text`

Writes a single `text` field containing role-tagged conversation text.

## Example

```bash
PYTHONPATH=src python3 -m kaggle_nvida.cli export-training \
  --manifest outputs/bootstrap/stage1_selection.jsonl \
  --output outputs/bootstrap/stage1_prompt_completion.jsonl \
  --format prompt_completion \
  --summary outputs/bootstrap/stage1_export_summary.json
```

## Notes

- the exporter resolves `path` and `line_number` from the manifest
- the first implementation uses a generic tagged-text format, not the final
  Nemotron production chat template
- once the exact training launcher is fixed, the exporter can add a dedicated
  template mode

