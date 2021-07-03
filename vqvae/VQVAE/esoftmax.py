import torch
import torch.nn.functional as F

INF = 1e6
EPS = 1e-6


def log_esoftmax(input: torch.Tensor, dim: int, training: bool = True) -> torch.Tensor:
    return torch.log(esoftmax(input, dim, training))

def esoftmax(input: torch.Tensor, dim: int, training: bool = False) -> torch.Tensor:
    mask = input < torch.mean(input, dim=dim, keepdim=True)
    mask_offset = torch.ones(input.shape, device=input.device, dtype=input.dtype)
    mask_offset[mask] = EPS if training else 0
    probs_unnormalized = F.softmax(input, dim=dim) * mask_offset
    probs = probs_unnormalized / torch.sum(probs_unnormalized, dim=dim, keepdim=True)
    return probs

def esoftmax_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "none",
    dim: int = -1,
    training: bool = True,
    ignore_index: int = -100
) -> torch.Tensor:
    return F.nll_loss(
        log_esoftmax(input, dim=dim, training=training),
        target,
        reduction=reduction,
        ignore_index=ignore_index,
    )


class LogESoftmax(torch.nn.Module):
    def __init__(self, dim: int = -1):
        super(LogESoftmax, self).__init__()
        self.dim = dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return log_esoftmax(X, self.dim, self.training)


class ESoftmax(torch.nn.Module):
    def __init__(self, dim: int = -1):
        super(ESoftmax, self).__init__()
        self.dim = dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return esoftmax(X, self.dim, self.training)


class ESoftmaxLoss(torch.nn.Module):
    def __init__(
        self, reduction: str = "none", dim: int = -1, ignore_index: int = -100
    ):
        super(ESoftmaxLoss, self).__init__()
        self.log_esoftmax = LogESoftmax(dim)
        self.reduction = reduction
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.nll_loss(
            self.log_esoftmax(input),
            target,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )
