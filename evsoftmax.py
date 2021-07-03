"""Implementation of ev-softmax function.

Authors: Phil Chen, Masha Itkina, Ransalu Senanayake, Mykel Kochenderfer
"""

import torch
import torch.nn.functional as F


INF = 1e6
EPS = 1e-6


def log_evsoftmax(input: torch.Tensor, dim: int, training: bool = True) -> torch.Tensor:
    return torch.log(evsoftmax(input, dim, training))

def evsoftmax(input: torch.Tensor, dim: int, training: bool = True) -> torch.Tensor:
    mask = input < torch.mean(input, dim=dim, keepdim=True)
    mask_offset = torch.ones(input.shape, device=input.device, dtype=input.dtype)
    mask_offset[mask] = EPS if training else 0
    probs_unnormalized = F.softmax(input, dim=dim) * mask_offset
    probs = probs_unnormalized / torch.sum(probs_unnormalized, dim=dim, keepdim=True)
    return probs

def evsoftmax_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "none",
    dim: int = -1,
    training: bool = True,
    ignore_index: int = -100
) -> torch.Tensor:
    return F.nll_loss(
        log_evsoftmax(input, dim=dim, training=training),
        target,
        reduction=reduction,
        ignore_index=ignore_index,
    )


class LogEvSoftmax(torch.nn.Module):
    def __init__(self, dim: int = -1):
        super(LogEvSoftmax, self).__init__()
        self.dim = dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return log_evsoftmax(X, self.dim, self.training)


class EvSoftmax(torch.nn.Module):
    def __init__(self, dim: int = -1):
        super(EvSoftmax, self).__init__()
        self.dim = dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return evsoftmax(X, self.dim, self.training)


class EvSoftmaxLoss(torch.nn.Module):
    def __init__(
        self, reduction: str = "none", dim: int = -1, ignore_index: int = -100
    ):
        super(EvSoftmaxLoss, self).__init__()
        self.log_evsoftmax = LogEvSoftmax(dim)
        self.reduction = reduction
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.nll_loss(
            self.log_evsoftmax(input),
            target,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )
