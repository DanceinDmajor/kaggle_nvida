import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kaggle_nvida.importers import import_jsonl_dataset
from kaggle_nvida.io_utils import read_jsonl


class ImporterTests(unittest.TestCase):
    def test_import_jsonl_dataset(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            summary = import_jsonl_dataset(
                input_path=Path("examples/importer/raw_instruction.jsonl"),
                mapping_path=Path("examples/importer/mapping_instruction.json"),
                output_path=output_dir / "train.jsonl",
                manifest_output_path=output_dir / "manifest.jsonl",
            )
            self.assertEqual(summary["input_rows"], 2)
            self.assertEqual(summary["dataset_rows"], 6)
            self.assertEqual(summary["manifest_rows"], 6)
            rows = list(read_jsonl(output_dir / "train.jsonl"))
            self.assertEqual(rows[0]["task_family"], "instruction")

