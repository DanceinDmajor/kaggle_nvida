import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kaggle_nvida.curation.mixture import build_stage_selection
from kaggle_nvida.exporters import export_training_dataset
from kaggle_nvida.pipeline import create_bootstrap_mixed_pack


class ExporterTests(unittest.TestCase):
    def test_export_prompt_completion(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mixed_dir = root / "mixed"
            create_bootstrap_mixed_pack(mixed_dir, count_per_family=2, seed=4)
            selection_path = root / "stage1_selection.jsonl"
            build_stage_selection(
                manifest_path=mixed_dir / "manifest.jsonl",
                config_path=Path("configs/mixtures/stage1_default.json"),
                output_path=selection_path,
                stage="stage1",
                max_examples=6,
            )
            output_path = root / "stage1_prompt_completion.jsonl"
            summary_path = root / "summary.json"
            summary = export_training_dataset(
                manifest_path=selection_path,
                output_path=output_path,
                format_name="prompt_completion",
                summary_path=summary_path,
            )
            self.assertEqual(summary["num_examples"], 6)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertIn("prompt", rows[0])
            self.assertIn("completion", rows[0])
            self.assertTrue(summary_path.exists())
