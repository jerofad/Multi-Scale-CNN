""" REF: """

import torch
from torch.nn.functional import cross_entropy, one_hot, softmax


class PolyLoss(torch.nn.Module):
    """ Implements PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    <https://arxiv.org/abs/2204.12511>

    for cross_entropy loss
    """

    def __init__(self, epsilon=2.0, reduction='none', weight=None):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    def forward(self, outputs, targets):
        ce = cross_entropy(outputs, targets, reduction='none', weight=self.weight)
        onehot_target = one_hot(targets)
        pt = torch.sum(onehot_target * softmax(outputs, dim=-1), dim=-1)
        poly =  ce + self.epsilon * (1.0 - pt)
        if self.reduction == 'mean':
            poly = poly.mean()
        elif self.reduction == 'sum':
            poly = poly.mean()

        return poly

