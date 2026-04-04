import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kaggle_nvida.curation.mixture import build_stage_selection
from kaggle_nvida.pipeline import create_bootstrap_mixed_pack
from kaggle_nvida.training import prepare_stage_run_bundle


class TrainingLauncherTests(unittest.TestCase):
    def test_prepare_stage_run_bundle(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mixed_dir = root / "mixed"
            create_bootstrap_mixed_pack(mixed_dir, count_per_family=2, seed=9)
            selection_path = root / "stage1_selection.jsonl"
            build_stage_selection(
                manifest_path=mixed_dir / "manifest.jsonl",
                config_path=Path("configs/mixtures/stage1_default.json"),
                output_path=selection_path,
                stage="stage1",
                max_examples=6,
            )

            run_dir = root / "runs" / "exp_stage1"
            summary = prepare_stage_run_bundle(
                stage="stage1",
                manifest_path=selection_path,
                run_dir=run_dir,
            )

            self.assertEqual(summary["stage"], "stage1")
            self.assertTrue((run_dir / "exports" / "stage1_train.chat.jsonl").exists())
            self.assertTrue((run_dir / "stage_config.json").exists())
            launch_plan = json.loads((run_dir / "launch_plan.json").read_text(encoding="utf-8"))
            self.assertIn("--config", launch_plan["launch_command"])
