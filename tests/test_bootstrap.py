import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kaggle_nvida.pipeline import create_bootstrap_math_pack


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_math_pack_generates_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            summary = create_bootstrap_math_pack(Path(temp_dir), count=3, seed=1)
            self.assertEqual(summary["dataset_rows"], 15)
            self.assertEqual(summary["manifest_rows"], 15)
