import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kaggle_nvida.io_utils import read_jsonl
from kaggle_nvida.pipeline import create_bootstrap_mixed_pack


class MixedBootstrapTests(unittest.TestCase):
    def test_bootstrap_mixed_pack_generates_multiple_families(self) -> None:
        with TemporaryDirectory() as temp_dir:
            summary = create_bootstrap_mixed_pack(Path(temp_dir), count_per_family=2, seed=3)
            self.assertEqual(summary["manifest_rows"], summary["dataset_rows"])
            rows = list(read_jsonl(Path(temp_dir) / "train.jsonl"))
            families = {row["task_family"] for row in rows}
            self.assertIn("math", families)
            self.assertIn("retrieval", families)
            self.assertIn("structured", families)
