import unittest

import torch

from mixformer.model import MixFormer


class MixFormerTests(unittest.TestCase):
    def test_padding_does_not_change_prediction(self):
        model = MixFormer(num_items=20, dim=16, heads=2, dropout=0).eval()
        candidate = torch.tensor([5])
        with torch.no_grad():
            short = model(torch.tensor([[1, 2, 3]]), candidate)
            padded = model(torch.tensor([[0, 0, 1, 2, 3]]), candidate)
        torch.testing.assert_close(short, padded)

    def test_batched_scores_and_backward(self):
        model = MixFormer(num_items=20, dim=16, heads=2, dropout=0)
        scores = model(torch.tensor([[1, 2, 0], [3, 4, 5]]), torch.tensor([6, 7]))
        self.assertEqual(tuple(scores.shape), (2,))
        scores.sum().backward()
        self.assertIsNotNone(model.item_embedding.weight.grad)


if __name__ == "__main__":
    unittest.main()
