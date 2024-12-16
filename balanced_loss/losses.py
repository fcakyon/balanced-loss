import numpy as np
import torch
import torch.nn.functional as F


def focal_loss(logits, labels, alpha=None, gamma=2):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bc_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bc_loss

    if alpha is not None:
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
    else:
        focal_loss = torch.sum(loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class Loss(torch.nn.Module):
    def __init__(
        self,
        loss_type: str = "cross_entropy",
        beta: float = 0.999,
        fl_gamma=2,
        samples_per_class=None,
        class_balanced=False,
        safe: bool = False,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.

        reference: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf

        Args:
            loss_type: string. One of "focal_loss", "cross_entropy",
                "binary_cross_entropy", "softmax_binary_cross_entropy".
            beta: float. Hyperparameter for Class balanced loss.
            fl_gamma: float. Hyperparameter for Focal loss.
            samples_per_class: A python list of size [num_classes].
                Required if class_balance is True.
            class_balanced: bool. Whether to use class balanced loss.
            safe: bool. Whether to allow labels with no samples.
        Returns:
            Loss instance
        """
        super(Loss, self).__init__()

        if class_balanced is True and samples_per_class is None:
            raise ValueError("samples_per_class cannot be None when class_balanced is True")

        self.loss_type = loss_type
        self.beta = beta
        self.fl_gamma = fl_gamma
        self.samples_per_class = samples_per_class
        self.class_balanced = class_balanced
        self.safe = safe

    def forward(self, logits: torch.tensor, labels: torch.tensor):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.

        Args:
            logits: A float tensor of size [batch, num_classes].
            labels: An int tensor of size [batch].
        Returns:
            cb_loss: A float tensor representing class balanced loss
        """

        batch_size = logits.size(0)
        num_classes = logits.size(1)
        labels_one_hot = F.one_hot(labels, num_classes).float()

        if self.class_balanced:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
            # Avoid division by 0 error for test cases without all labels present.
            if self.safe:
                effective_num_classes = np.sum(effective_num != 0)
                effective_num[effective_num == 0] = np.inf

            else:
                effective_num_classes = num_classes

            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * effective_num_classes
            weights = torch.tensor(weights, device=logits.device).float()

            if self.loss_type != "cross_entropy":
                weights = weights.unsqueeze(0)
                weights = weights.repeat(batch_size, 1) * labels_one_hot
                weights = weights.sum(1)
                weights = weights.unsqueeze(1)
                weights = weights.repeat(1, num_classes)
        else:
            weights = None

        if self.loss_type == "focal_loss":
            cb_loss = focal_loss(logits, labels_one_hot, alpha=weights, gamma=self.fl_gamma)
        elif self.loss_type == "cross_entropy":
            cb_loss = F.cross_entropy(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "binary_cross_entropy":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "softmax_binary_cross_entropy":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss
