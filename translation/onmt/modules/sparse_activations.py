"""
An implementation of sparsemax (Martins & Astudillo, 2016). See
:cite:`DBLP:journals/corr/MartinsA16` for detailed description.

By Ben Peters and Vlad Niculae
"""

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
import torch.nn as nn
import torch.nn.functional as F


INF = 1e6
EPS = 1e-6

def evsoftmax(input: torch.Tensor, dim: int, training: bool = True) -> torch.Tensor:
    mask = input < torch.mean(input, dim=dim, keepdim=True)
    mask_offset = torch.ones(input.shape, device=input.device, dtype=input.dtype)
    mask_offset[mask] = EPS if training else 0
    probs_unnormalized = F.softmax(input, dim=dim) * mask_offset
    probs = probs_unnormalized / torch.sum(probs_unnormalized, dim=dim, keepdim=True)
    return probs


def log_evsoftmax(input: torch.Tensor, dim: int, training: bool = True) -> torch.Tensor:
    return torch.log(evsoftmax(input, dim, training))


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


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _threshold_and_support(input, dim=0):
    """Sparsemax building block: compute the threshold

    Args:
        input: any dimension
        dim: dimension along which to apply the sparsemax

    Returns:
        the threshold value
    """

    input_srt, _ = torch.sort(input, descending=True, dim=dim)
    input_cumsum = input_srt.cumsum(dim) - 1
    rhos = _make_ix_like(input, dim)
    support = rhos * input_srt > input_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)
    return tau, support_size


class SparsemaxFunction(Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input, dim=0):
        """sparsemax: normalizing sparse transform (a la softmax)

        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax

        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = _threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):

    def __init__(self, dim=0):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


class LogSparsemax(nn.Module):

    def __init__(self, dim=0):
        self.dim = dim
        super(LogSparsemax, self).__init__()

    def forward(self, input):
        return torch.log(sparsemax(input, self.dim))


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _roll_last(X, dim):
    if dim == -1:
        return X
    elif dim < 0:
        dim = X.dim() - dim

    perm = [i for i in range(X.dim()) if i != dim] + [dim]
    return X.permute(perm)


def _tsallis_threshold_and_support(input, dim=0):
    Xsrt, _ = torch.sort(input, descending=True, dim=dim)

    rho = _make_ix_like(input, dim)
    mean = Xsrt.cumsum(dim) / rho
    mean_sq = (Xsrt ** 2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean ** 2)
    delta = (1 - ss) / rho

    # NOTE this is not exactly the same as in reference algo
    # Fortunately it seems the clamped values never wrongly
    # get selected by tau <= sorted_z. Prove this!
    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)

    support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
    tau_star = tau.gather(dim, support_size - 1)
    return tau_star, support_size


def _tsallis_threshold_and_support_topk(input, dim=0, k=100):

    if k >= input.shape[dim]:  # do full sort
        Xsrt, _ = torch.sort(input, dim=dim, descending=True)
    else:
        Xsrt, _ = torch.topk(input, k=k, dim=dim)

    rho = _make_ix_like(Xsrt, dim)
    mean = Xsrt.cumsum(dim) / rho
    mean_sq = (Xsrt ** 2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean ** 2)
    delta = (1 - ss) / rho

    # NOTE this is not exactly the same as in reference algo
    # Fortunately it seems the clamped values never wrongly
    # get selected by tau <= sorted_z. Prove this!
    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)

    support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
    tau_star = tau.gather(dim, support_size - 1)

    unsolved = (support_size == k).squeeze(dim)

    if torch.any(unsolved):
        X_ = _roll_last(input, dim)[unsolved]
        tau_, ss_ = _tsallis_threshold_and_support_topk(X_, dim=-1, k=2 * k)
        _roll_last(tau_star, dim)[unsolved] = tau_
        _roll_last(support_size, dim)[unsolved] = ss_

    return tau_star, support_size


class Tsallis15Function(Function):
    @staticmethod
    def forward(ctx, X, dim=0):
        ctx.dim = dim

        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as for softmax
        X = X / 2  # divide by 2 to solve actual Tsallis

        tau_star, _ = _tsallis_threshold_and_support(X, dim)

        Y = torch.clamp(X - tau_star, min=0) ** 2
        ctx.save_for_backward(Y)
        return Y

    @staticmethod
    def backward(ctx, dY):
        (Y,) = ctx.saved_tensors
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None


class Tsallis15TopKFunction(Tsallis15Function):
    @staticmethod
    def forward(ctx, X, dim=0, k=100):
        ctx.dim = dim

        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as for softmax
        X = X / 2  # divide by 2 to solve actual Tsallis

        tau_star, _ = _tsallis_threshold_and_support_topk(X, dim=dim, k=k)

        Y = torch.clamp(X - tau_star, min=0) ** 2
        ctx.save_for_backward(Y)
        return Y

    @staticmethod
    def backward(ctx, dY):
        return Tsallis15Function.backward(ctx, dY) + (None,)


tsallis15 = Tsallis15Function.apply
tsallis15_topk = Tsallis15TopKFunction.apply


class Tsallis15(torch.nn.Module):
    def __init__(self, dim=0):
        self.dim = dim
        super(Tsallis15, self).__init__()

    def forward(self, X):
        return tsallis15(X, self.dim)


class LogTsallis15(torch.nn.Module):
    def __init__(self, dim=0):
        self.dim = dim
        super(LogTsallis15, self).__init__()

    def forward(self, X):
        return torch.log(tsallis15(X, self.dim))
