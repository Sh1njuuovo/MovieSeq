import tempfile
import unittest
from pathlib import Path

from mixformer.prepare import prepare_movielens


class PrepareTests(unittest.TestCase):
    def test_chronological_examples_with_negative_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            ratings = Path(folder) / "ratings.dat"
            ratings.write_text("1::10::5::1\n1::20::4::2\n1::30::5::3\n2::40::5::1\n2::10::5::2\n")
            rows = prepare_movielens(ratings, min_rating=4, negatives=1)
        self.assertTrue(rows)
        self.assertEqual(rows[0]["history"], [1])
        self.assertIn({row["label"] for row in rows}, ({0, 1},))
