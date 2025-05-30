import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef, f1_score

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias = False)
        # split the weight update component to direction and norm
        # WeightNorm.apply(self.L, 'weight', dim=0)

        # a fixed scale factor to scale the output of cos value
        # into a reasonably large input for softmax
        self.scale_factor = 10

    def forward(self, x):

        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        # L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)

        # self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)

        cos_dist = self.L(x_normalized)  # matrix product by forward function
        scores = self.scale_factor * (cos_dist)

        return scores
    
class BASE(nn.Module):
    '''
        BASE model
    '''
    def __init__(self, args):
        super(BASE, self).__init__()
        self.args = args

        # cached tensor for speed
        # self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float),
        #                           requires_grad=False)
    
    @staticmethod
    def compute_acc(pred, true, dim=1, nomax=False):
        '''
            Compute the accuracy.
            @param pred: batch_size * num_classes
            @param true: batch_size
        '''
        if nomax: return torch.mean((pred == true).float()).item()
        else: return torch.mean((torch.argmax(pred, dim=dim) == true).float()).item()
        
    @staticmethod
    def compute_f1(y_pred, true, dim=1, nomax=False,  labels=None, average='weighted'):
        '''
            Compute the weighted f1 score.
            @param pred: batch_size * num_classes
            @param true: batch_size
        '''
        if not nomax: _, y_pred = torch.max(y_pred, dim)

        f1 = f1_score(true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average=average, labels=labels )

        return f1

    @staticmethod
    def compute_mcc(y_pred, true, dim=1, nomax=False):
        '''
            Compute the matthews correlation coeficient.
            @param pred: batch_size * num_classes
            @param true: batch_size
        '''
        if not nomax: _, y_pred = torch.max(y_pred, dim)

        mcc = matthews_corrcoef(true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

        return mcc