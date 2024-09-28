import torch
import numpy as np
import torch.nn as nn

def get_kld_loss(mu, logvar, dim=1, is_mean=False):
    # from https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=dim)
    #
    if is_mean:
        KLD = torch.mean(KLD * -0.5)
    else:
        KLD = torch.sum(KLD) * -0.5
    return KLD


# From https://github.com/DenisDsh/PyTorch-Deep-CORAL/blob/master/coral.py
def coral(source, target):

    d = source.size(1)  # dim vector

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    loss = loss / (4 * d * d)
    return loss


def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c

class ZINORMLoss(nn.Module):
    def __init__(self, reduction='none', eps=1e-10):
        super().__init__()
        self.eps = eps
        self.scale_factor = 1.0
        self.reduction = reduction
        self.constant_pi = torch.acos(torch.zeros(1)).item() * 2


    def forward(self, y_pred, theta, pi, y_true, lamda, mask=None):
        if mask is not None:
            y_true = torch.masked_select(y_true, mask)
            y_pred = torch.masked_select(y_pred, mask)
            theta = torch.masked_select(theta, mask)
            pi = torch.masked_select(pi, mask)

        zero_case = self.zero_case_loss(y_pred, theta, pi, y_true)
        norm_case = self.norm_case_loss(y_pred, theta, pi, y_true)
        t1 = torch.where(torch.less(y_true, 1e-8), zero_case, norm_case)
        t2 = lamda * torch.square(pi)
        loss = t1 + t2
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def norm_case_loss(self, y_pred, theta, pi, y_true):
        theta = torch.clamp(theta, min=self.eps, max=1e6)
        t1 = -torch.log(1.0 - pi + self.eps)
        t2 = -0.5 * torch.log(2.0 * self.constant_pi * theta) - \
            torch.square(y_true - y_pred) / ((2 * theta))
        norm_case = t1 - t2
        return norm_case

    def zero_case_loss(self, y_pred, theta, pi, y_true):
        theta = torch.clamp(theta, min=self.eps, max=1e6)
        zero_norm = 1.0 / torch.sqrt(2.0 * self.constant_pi * theta + self.eps) * torch.exp(-0.5 * (
            (0. - y_pred) ** 2) / theta + self.eps)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_norm) + self.eps)
        return zero_case

