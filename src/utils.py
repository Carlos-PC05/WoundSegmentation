import torch
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        #A is predictions, B is the target

        #Transforms the values to [0,1]
        pred = torch.sigmoid(pred)

        #formula dice coefficient -> (2 * intersection between A and B) / (A + B)
        num = 2 * (pred * target).sum()
        den = pred.sum() + target.sum()

        #To avoid dividing by 0
        smooth = 1e-6

        num += smooth
        den += smooth

        #Dice Loss = 1 - Dice coefficient
        return 1 - (num / den)
    

