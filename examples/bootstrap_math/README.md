# Bootstrap Math Example

Generate a tiny starter dataset:

```bash
PYTHONPATH=src python3 -m kaggle_nvida.cli bootstrap-math \
  --output-dir examples/bootstrap_math/generated \
  --count 5
```

This produces:

- `train.jsonl`
- `manifest.jsonl`

