import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kaggle_nvida.integrations import materialize_tool_stack_bundle


class ToolStackTests(unittest.TestCase):
    def test_materialize_tool_stack_bundle(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "selection.jsonl"
            export_path = root / "train.jsonl"
            eval_slice_dir = root / "eval_slices"
            bundle_dir = root / "tool_stack"

            manifest_path.write_text("", encoding="utf-8")
            export_path.write_text("", encoding="utf-8")
            eval_slice_dir.mkdir(parents=True, exist_ok=True)

            summary = materialize_tool_stack_bundle(
                stage="stage2",
                train_manifest_path=manifest_path,
                export_path=export_path,
                eval_slice_dir=eval_slice_dir,
                output_dir=bundle_dir,
            )

            self.assertEqual(summary["stage"], "stage2")
            recipe_path = bundle_dir / "nemo_rl_recipe.json"
            recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            self.assertEqual(recipe["objective"], "sft_hardening")
            self.assertEqual(recipe["inputs"]["train_manifest"], str(manifest_path))
