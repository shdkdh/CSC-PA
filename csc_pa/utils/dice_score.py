import torch
from torch.autograd import Function

def dice_coeff2(pred, gt, smooth=1e-5):
    """ computational formula：
        dice = 2TP/(FP + 2TP + FN)
    """
    N = gt.shape[0]
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    dice = (2 * intersection + smooth) / (unionset + smooth)
    return dice.sum() / N

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)

        eps = 0.0001

        self.inter = torch.sum(input.view(-1) * target.view(-1))
        self.union = torch.sum(input) + torch.sum(target)

        t = (2 * self.inter.float() + eps) / (self.union.float() + eps)
        return t

    def backward(self, grad_output):
        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) / (self.union * self.union)

        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

def dice_coeff(input, target, device):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s/(i+1)