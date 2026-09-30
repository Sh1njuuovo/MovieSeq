import json
import tempfile
import unittest
from pathlib import Path

from mixformer.train import train_file


class TrainingTests(unittest.TestCase):
    def test_one_epoch_on_small_dataset(self):
        rows = [
            {"history": [1, 2], "candidate": 3, "label": 1},
            {"history": [1, 3], "candidate": 4, "label": 0},
            {"history": [2, 4], "candidate": 5, "label": 1},
            {"history": [2, 5], "candidate": 1, "label": 0},
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "examples.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            report = train_file(path, epochs=1, dim=8, heads=2, batch_size=2)
        self.assertEqual(report["examples"], 4)
        self.assertEqual(len(report["epochs"]), 1)
