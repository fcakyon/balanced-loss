<div align="center">
<h2>
    Balanced Loss
</h2>
</div>

<div align="center">
    <a href="https://badge.fury.io/py/balanced-loss"><img src="https://badge.fury.io/py/balanced-loss.svg" alt="pypi version"></a>

    Easy to use class balanced cross entropy and focal loss implementation for Pytorch.
</div>



<p align="center">
<img src="https://user-images.githubusercontent.com/34196005/180311379-1003da44-cdf9-46e8-af83-e65fbc3710cd.png" width="600">
</p>

## Theory

When training dataset labels are imbalanced, one thing to do is to balance the loss across sample classes.

- First, the effective number of samples are calculated for all classes as:

![alt-text](https://user-images.githubusercontent.com/34196005/180266195-aa2e8696-cdeb-48ed-a85f-7ffb353942a4.png)

- Then the class balanced loss function is defined as:

![alt-text](https://user-images.githubusercontent.com/34196005/180266198-e27d8cba-f5e1-49ca-9f82-d8656333e3c4.png)


## Installation

```bash
pip install balanced-loss
```

## Usage

- Standard losses:

```python
import torch
from balanced_loss import Loss

# outputs and labels
logits = torch.tensor([[0.78, 0.1, 0.05]]) # 1 batch, 3 class
labels = torch.tensor([0]) # 1 batch

# focal loss
focal_loss = Loss(loss_type="focal_loss")
loss = focal_loss(logits, labels)

# cross-entropy loss
ce_loss = Loss(loss_type="cross_entropy")
loss = ce_loss(logits, labels)

# binary cross-entropy loss
bce_loss = Loss(loss_type="binary_cross_entropy")
loss = bce_loss(logits, labels)
```

- Class-balanced losses:

```python
import torch
from balanced_loss import Loss

# outputs and labels
logits = torch.tensor([[0.78, 0.1, 0.05]]) # 1 batch, 3 class
labels = torch.tensor([0]) # 1 batch

# number of samples per class in the training dataset
samples_per_class = [30, 100, 25] # 30, 100, 25 samples for labels 0, 1 and 2, respectively

# class-balanced focal loss
focal_loss = Loss(
    loss_type="focal_loss",
    samples_per_class=samples_per_class,
    class_balanced=True
)
loss = focal_loss(logits, labels)

# class-balanced cross-entropy loss
ce_loss = Loss(
    loss_type="cross_entropy",
    samples_per_class=samples_per_class,
    class_balanced=True
)
loss = ce_loss(logits, labels)

# class-balanced binary cross-entropy loss
bce_loss = Loss(
    loss_type="binary_cross_entropy",
    samples_per_class=samples_per_class,
    class_balanced=True
)
loss = bce_loss(logits, labels)
```

- Customize parameters:

```python
import torch
from balanced_loss import Loss

# outputs and labels
logits = torch.tensor([[0.78, 0.1, 0.05]]) # 1 batch, 3 class
labels = torch.tensor([0])

# number of samples per class in the training dataset
samples_per_class = [30, 100, 25] # 30, 100, 25 samples for labels 0, 1 and 2, respectively

# class-balanced focal loss
focal_loss = Loss(
    loss_type="focal_loss",
    beta=0.999, # class-balanced loss beta
    fl_gamma=2, # focal loss gamma
    samples_per_class=samples_per_class,
    class_balanced=True
)
loss = focal_loss(logits, labels)
```

## Improvements

What is the difference between this repo and vandit15's?

- This repo is a pypi installable package
- This repo implements loss functions as `torch.nn.Module`
- In addition to class balanced losses, this repo also supports the standard versions of the cross entropy/focal loss etc. over the same API
- All typos and errors in vandit15's source are fixed

## References

https://arxiv.org/abs/1901.05555

https://github.com/richardaecn/class-balanced-loss

https://github.com/vandit15/Class-balanced-loss-pytorch