import unittest


class TestLosses(unittest.TestCase):
    def test_standard_losses(self):
        import torch

        from balanced_loss import Loss

        # outputs and labels
        logits = torch.tensor([[0.78, 0.1, 0.05], [0.78, 0.83, 0.05]])  # 2 batch, 3 class
        labels = torch.tensor([0, 2])  # 2 batch

        # focal loss
        focal_loss = Loss(loss_type="focal_loss")
        loss = focal_loss(logits, labels)

        self.assertAlmostEqual(loss.item(), 0.85, delta=0.01)

        # cross-entropy loss
        ce_loss = Loss(loss_type="cross_entropy")
        loss = ce_loss(logits, labels)

        self.assertAlmostEqual(loss.item(), 1.17, delta=0.01)

        # binary cross-entropy loss
        bce_loss = Loss(loss_type="binary_cross_entropy")
        loss = bce_loss(logits, labels)

        self.assertAlmostEqual(loss.item(), 0.80, delta=0.01)

    def test_balanced_losses(self):
        import torch

        from balanced_loss import Loss

        # outputs and labels
        logits = torch.tensor([[0.78, 0.1, 0.05], [0.78, 0.83, 0.05]])  # 2 batch, 3 class
        labels = torch.tensor([0, 2])  # 2 batch

        # number of samples per class in the training dataset
        samples_per_class = [30, 100, 25]  # 30, 100, 25 samples for labels 0, 1 and 2, respectively

        # class-balanced focal loss
        focal_loss = Loss(loss_type="focal_loss", samples_per_class=samples_per_class, class_balanced=True)
        loss = focal_loss(logits, labels)

        self.assertAlmostEqual(loss.item(), 1.17, delta=0.01)

        # class-balanced cross-entropy loss
        ce_loss = Loss(loss_type="cross_entropy", samples_per_class=samples_per_class, class_balanced=True)
        loss = ce_loss(logits, labels)

        self.assertAlmostEqual(loss.item(), 1.59, delta=0.01)

        # class-balanced binary cross-entropy loss
        bce_loss = Loss(loss_type="binary_cross_entropy", samples_per_class=samples_per_class, class_balanced=True)
        loss = bce_loss(logits, labels)

        self.assertAlmostEqual(loss.item(), 1.08, delta=0.01)


if __name__ == "__main__":
    unittest.main()
