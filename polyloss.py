""" REF: """

import torch
from torch.nn.functional import cross_entropy, one_hot, softmax


class PolyLoss(torch.nn.Module):
    """ Implements PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    <https://arxiv.org/abs/2204.12511>

    for cross_entropy loss
    """

    def __init__(self, epsilon=1.0, reduction='none',num_classes=3, weight=None):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        ce = cross_entropy(outputs, targets, reduction='none', weight=self.weight)
        # pt = one_hot(, outputs.size()[1]) * softmax(outputs, 1)
        onehot_target = one_hot(targets, num_classes=self.num_classes)
        
        pt = torch.sum(onehot_target * softmax(outputs, dim=-1), dim=-1)
        poly =  ce + self.epsilon * (1.0 - pt)
        if self.reduction == 'mean':
            poly = poly.mean()
        elif self.reduction == 'sum':
            poly = poly.mean()

        return poly

