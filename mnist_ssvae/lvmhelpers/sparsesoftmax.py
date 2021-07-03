import torch
import torch.nn.functional as F

INF = 1e6

def sparse_softmax(X, dim: int = -1, train: bool = True):
    mask = X < torch.mean(X, dim=dim, keepdim=True)
    mask_offset = mask * (INF if train else float("Inf"))
    probs = F.softmax(X - mask_offset, dim=dim)
    return probs


class LogSparseSoftmax(torch.nn.Module):
    def __init__(self, dim: int = -1, train: bool = True):
        super(LogSparseSoftmax, self).__init__()
        self.dim = dim
        self.train = train
    
    def forward(self, X):
        mask = X < torch.mean(X, dim=self.dim, keepdim=True)
        mask_offset = mask * (INF if self.train else float("Inf"))
        log_probs = F.log_softmax(X - mask_offset, dim=self.dim)
        return log_probs


class SparseSoftmax(torch.nn.Module):
    def __init__(self, dim: int = -1, train: bool = True):
        super(SparseSoftmax, self).__init__()
        self.dim = dim
        self.train = train
    
    def forward(self, X):
        mask = X < torch.mean(X, dim=self.dim, keepdim=True)
        mask_offset = mask * (INF if self.train else float("Inf"))
        probs = F.softmax(X - mask_offset, dim=self.dim)
        return probs


class SparseSoftmaxLoss(torch.nn.Module):
    def __init__(self, reduction: str = 'none', dim: int = -1):
        super(SparseSoftmaxLoss, self).__init__()
        self.log_sparse_softmax = LogSparseSoftmax(dim)
        self.reduction = reduction
        self.dim = dim
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return F.nll_loss(self.log_sparse_softmax(input), target, reduction=self.reduction)
