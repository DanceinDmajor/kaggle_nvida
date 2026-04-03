import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kaggle_nvida.tracking import init_experiment_run, record_experiment_result


class TrackerTests(unittest.TestCase):
    def test_init_run_and_record_result(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir) / "runs"
            run_dir = init_experiment_run(runs_dir, "exp_demo")
            self.assertTrue(run_dir.exists())
            tracker_path = Path(temp_dir) / "experiment_tracker.csv"
            record_experiment_result(
                tracker_path,
                {
                    "experiment_id": "exp_demo",
                    "objective": "stage1_sft",
                    "data_mix": "stage1_default",
                },
            )
            self.assertTrue(tracker_path.exists())
