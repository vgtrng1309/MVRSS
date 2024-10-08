from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .one_hot import one_hot


# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

def log_cosh_soft_dice_loss(input: torch.Tensor, target: torch.Tensor, eps: float = 1e-8,
                   global_weight: float = 1.) -> torch.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.
    Arthur: add ^2 for soft formulation

    See :class:`~kornia.losses.DiceLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                         .format(input.shape))

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError("input and target shapes must be the same. Got: {} and {}"
                         .format(input.shape, input.shape))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(torch.pow(input_soft, 2) + torch.pow(target_one_hot, 2), dims)

    dice_score = 2. * intersection / (cardinality + eps)
    return global_weight*torch.mean(torch.log(torch.cosh(-dice_score + 1.)))


class LogCoshSoftDiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.
    Arthur: add ^2 for soft formulation

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, global_weight: float = 1.) -> None:
        super(LogCoshSoftDiceLoss, self).__init__()
        self.eps: float = 1e-6
        self.global_weight = global_weight

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        return log_cosh_soft_dice_loss(input, target, self.eps, self.global_weight)

