import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from onmt.modules.sparse_activations import (
    _threshold_and_support, log_evsoftmax, LogEvSoftmax, tsallis15
)
from onmt.utils.misc import aeq


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


class SparsemaxLossFunction(Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input, target):
        """
        input (FloatTensor): ``(n, num_classes)``.
        target (LongTensor): ``(n,)``, the indices of the target classes
        """
        input_batch, classes = input.size()
        target_batch = target.size(0)
        aeq(input_batch, target_batch)

        z_k = input.gather(1, target.unsqueeze(1)).squeeze()
        tau_z, support_size = _threshold_and_support(input, dim=1)
        support = input > tau_z
        x = torch.where(
            support, input**2 - tau_z**2,
            torch.tensor(0.0, device=input.device)
        ).sum(dim=1)
        ctx.save_for_backward(input, target, tau_z)
        # clamping necessary because of numerical errors: loss should be lower
        # bounded by zero, but negative values near zero are possible without
        # the clamp
        return torch.clamp(x / 2 - z_k + 0.5, min=0.0)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, target, tau_z = ctx.saved_tensors
        sparsemax_out = torch.clamp(input - tau_z, min=0)
        delta = torch.zeros_like(sparsemax_out)
        delta.scatter_(1, target.unsqueeze(1), 1)
        return sparsemax_out - delta, None


sparsemax_loss = SparsemaxLossFunction.apply

def _fy_backward(ctx, grad_output):
    (p_star,) = ctx.saved_tensors
    grad = grad_output.unsqueeze(1) * p_star
    return grad


# more efficient specializations
def _omega_tsallis15(p_star):
    return (1 - (p_star * torch.sqrt(p_star)).sum(dim=1)) / 0.75


class SparsemaxLoss(nn.Module):
    """
    An implementation of sparsemax loss, first proposed in
    :cite:`DBLP:journals/corr/MartinsA16`. If using
    a sparse output layer, it is not possible to use negative log likelihood
    because the loss is infinite in the case the target is assigned zero
    probability. Inputs to SparsemaxLoss are arbitrary dense real-valued
    vectors (like in nn.CrossEntropyLoss), not probability vectors (like in
    nn.NLLLoss).
    """

    def __init__(self, weight=None, ignore_index=-100,
                 reduction='elementwise_mean'):
        assert reduction in ['elementwise_mean', 'sum', 'none']
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        super(SparsemaxLoss, self).__init__()

    def forward(self, input, target):
        loss = sparsemax_loss(input, target)
        if self.ignore_index >= 0:
            ignored_positions = target == self.ignore_index
            size = float((target.size(0) - ignored_positions.sum()).item())
            loss.masked_fill_(ignored_positions, 0.0)
        else:
            size = float(target.size(0))
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'elementwise_mean':
            loss = loss.sum() / size
        return loss


class Tsallis15LossFunction(Function):
    @staticmethod
    def forward(ctx, input, target):
        """
        input (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """

        p_star = tsallis15(input, 1)
        loss = _omega_tsallis15(p_star)

        p_star.scatter_add_(1, target.unsqueeze(1), torch.full_like(p_star, -1))
        loss += torch.einsum("ij,ij->i", p_star, input)

        ctx.save_for_backward(p_star)

        # loss = torch.clamp(loss, min=0.0)  # needed?
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        return _fy_backward(ctx, grad_output), None

