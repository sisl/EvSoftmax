# Code modified from: https://github.com/timbmg/VAE-CVAE-MNIST/

seed = 1243

import numpy as np
np.random.seed(seed)
import torch
import torch.nn.functional as F

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

from torch.autograd import Variable

INF = 1e6

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


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def idx2onehot(idx, n):

    assert idx.size(1) == 1
    assert torch.max(idx).data < n

    onehot = torch.zeros(idx.size(0), n).cuda()
    onehot.scatter_(1, idx.data, 1)
    onehot = to_var(onehot)
    
    return onehot

# Function from: https://github.com/EmilienDupont/vae-concrete/
def kl_discrete(alpha, num_classes = 10, epsilon=1e-8):
    """
    KL divergence between a uniform distribution over num_cat categories and
    dist.
    Parameters
    ----------
    alpha : Tensor - shape (None, num_categories)
    num_cat : int
    """
    alpha_sum = torch.sum(alpha, axis=1)  # Sum over columns, this now has size (batch_size,)
    alpha_neg_entropy = torch.sum(alpha * torch.log(alpha + epsilon), axis=1)
    return np.log(num_classes) + torch.mean(alpha_neg_entropy - alpha_sum)

def kl_q_p(q_dist, p_dist):
    kl_separated = td.kl_divergence(self.q_dist, self.p_dist)
    if len(kl_separated.size()) < 2:
        kl_separated = torch.unsqueeze(kl_separated, dim=0)
        
    kl_minibatch = torch.mean(kl_separated, dim=0, keepdim=True)
    kl = torch.sum(kl_minibatch)
        
    return kl

def sample_q(alpha, train=True, temperature=0.67):
	
    if train:
        zdist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(torch.tensor([temperature]).cuda(), probs = alpha)
        sample = zdist.rsample()
    else:
        zdist = torch.distributions.one_hot_categorical.OneHotCategorical(probs=alpha)
        sample = zdist.sample()

    return sample

def sample_p(alpha, batch_size=1):
    zdist = torch.distributions.one_hot_categorical.OneHotCategorical(probs = alpha)
    return zdist.sample(torch.Size([batch_size]))