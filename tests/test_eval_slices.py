import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kaggle_nvida.curation.mixture import build_stage_selection
from kaggle_nvida.datasets import resolve_manifest_examples
from kaggle_nvida.evaluation import build_eval_slices, score_prediction_file
from kaggle_nvida.pipeline import create_bootstrap_mixed_pack


class EvalSliceTests(unittest.TestCase):
    def test_build_eval_slices_and_score_predictions(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mixed_dir = root / "mixed"
            create_bootstrap_mixed_pack(mixed_dir, count_per_family=3, seed=12)
            selection_path = root / "stage1_selection.jsonl"
            build_stage_selection(
                manifest_path=mixed_dir / "manifest.jsonl",
                config_path=Path("configs/mixtures/stage1_default.json"),
                output_path=selection_path,
                stage="stage1",
                max_examples=9,
            )

            eval_dir = root / "eval_slices"
            summary = build_eval_slices(selection_path, eval_dir)
            self.assertTrue((eval_dir / "overall_top.jsonl").exists())
            self.assertTrue(summary["slices"])

            examples = resolve_manifest_examples(eval_dir / "overall_top.jsonl")
            predictions_path = root / "predictions.jsonl"
            predictions = []
            for example in examples:
                predictions.append(
                    {
                        "example_id": example["example_id"],
                        "prediction": f"Final answer: {example['target']['final_answer']}",
                    }
                )
            predictions_path.write_text(
                "\n".join(json.dumps(row) for row in predictions) + "\n",
                encoding="utf-8",
            )

            score = score_prediction_file(
                slice_manifest_path=eval_dir / "overall_top.jsonl",
                predictions_path=predictions_path,
            )
            self.assertEqual(score["exact_match"], 1.0)
            self.assertEqual(score["format_adherence"], 1.0)
