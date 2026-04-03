# Bootstrap Mixed Example

Generate a broader starter dataset:

```bash
PYTHONPATH=src python3 -m kaggle_nvida.cli bootstrap-mixed \
  --output-dir examples/bootstrap_mixed/generated \
  --count-per-family 5
```

This produces:

- `train.jsonl`
- `manifest.jsonl`

