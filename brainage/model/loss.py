import torch
from torch.nn import functional as F

class class_loss(torch.nn.Module):

    def __init__(self, label_range=[20.0, 80.0], label_step=2.5):
        super().__init__()
        self.label_range = label_range
        self.label_step  = label_step
    
    def forward(self, y_hat, y):
        y_c = ((y.clamp(self.label_range[1] - self.label_range[0])-self.label_range[0])//self.label_step).long()
        loss = F.cross_entropy(y_hat, y_c)
        return loss, (torch.argmax(y_hat, dim=1)*self.label_step+self.label_range[0]).float()


class l2_loss(torch.nn.Module):
    
    def __init__(self, heteroscedastic=False):
        super().__init__()
        self.heteroscedastic = heteroscedastic
        
    def forward(self, y_hat, y):
        if self.heteroscedastic:
            mu = y_hat[:, 0]
            log_sigma = y_hat[:, 1]
            loss = torch.sum(0.5*(torch.exp(log_sigma)**(-2))*(mu-y)**2 + log_sigma)
        else:
            loss = F.mse_loss(y_hat[:, 0], y)
        return loss, y_hat[:, 0]