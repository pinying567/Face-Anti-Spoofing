import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma
        self.criterion = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs.squeeze(), targets.float())
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class MCFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(MCFocalLoss, self).__init__()
        self.alpha = torch.tensor([0.5, 0.25, 0.25]).cuda() # weight for real/print/replay
        self.gamma = gamma

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.data.view(-1, 1))
        logpt = logpt.view(-1)
        pt = logpt.data.exp()
        at = self.alpha.gather(0, targets.data.view(-1))
        F_loss = -at * (1-pt)**self.gamma * logpt
        return F_loss.mean()

