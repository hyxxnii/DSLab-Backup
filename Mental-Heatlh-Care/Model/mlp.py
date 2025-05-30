import torch
import torch.nn as nn
from .base import BASE

class MLPseq(BASE):
    def __init__(self, ebd_dim, args, top_layer=None):
        super(MLPseq, self).__init__(args)

        self.args = args
        self.ebd_dim = ebd_dim

        self.mlp = self._init_mlp(ebd_dim, self.args.mlp_hidden, self.args.dropout)
        self.out = self.get_top_layer(self.args, self.args.n_classes)
        #self.top_layer = top_layer
        self.dropout = nn.Dropout(self.args.dropout)

    @staticmethod
    def get_top_layer(args, n_classes):
        '''
            Creates final layer of desired type
            @return final classification layer
        '''
        #loss_type = args.finetune_loss_type

        #if loss_type == 'softmax':
        return nn.Linear(args.mlp_hidden[-1], n_classes)
        # elif loss_type == 'dist':
        #     return distLinear(args.mlp_hidden[-1], n_classes)
        
    def _init_mlp(self, in_d, hidden_ds, drop_rate):
        modules = []

        for d in hidden_ds[:-1]:
            modules.extend([
                nn.Dropout(drop_rate),
                nn.Linear(in_d, d),
                nn.ReLU()])
            in_d = d

        modules.extend([
            nn.Dropout(drop_rate),
            nn.Linear(in_d, hidden_ds[-1])])

        return nn.Sequential(*modules)
    
    def forward(self, XS, YS=None, XQ=None, YQ=None, weights=None, return_preds=False):
        '''
            if y is specified, return loss and accuracy
            otherwise, return the transformed x

            @param: XS: batch_size * input_dim
            @param: YS: batch_size (optional)

            @return: XS: batch_size * output_dim
        '''

        # normal training procedure, train stage only use query
        # if weights is None:
        #     XS = self.mlp(XS)
        # else:
        #     # find weight and bias keys for the mlp module
        #     w_keys, b_keys = [], []
        #     for key in weights.keys():
        #         if key[:4] == 'mlp.':
        #             if key[-6:] == 'weight':
        #                 w_keys.append(key)
        #             else:
        #                 b_keys.append(key)

        #     for i in range(len(w_keys)-1):
        #         #XS = F.dropout(XS, self.args.dropout, training=self.training)
        #         XS = self.dropout(XS)
        #         XS = F.linear(XS, weights[w_keys[i]], weights[b_keys[i]])
        #         XS = F.relu(XS)

        #     XS = F.dropout(XS, self.args.dropout, training=self.training)
        #     XS = F.linear(XS, weights[w_keys[-1]], weights[b_keys[-1]])

        XS = self.mlp(XS)
        XS = self.out(XS) # output: [batch, max_sentence, n_class]
        
        # if self.top_layer is not None:
        #     XS = self.top_layer(XS)

        # # normal training procedure, compute loss/acc
        # if YS is not None:
        #     # if self.args.taskmode == 'episodic':
        #     #     ## useful for episodes, ignored for full supervised
        #     #     _, YS = torch.unique(YS, sorted=True, return_inverse=True)
        #     loss = F.cross_entropy(XS, YS)
        #     acc = BASE.compute_acc(XS, YS)
        #     f1 = BASE.compute_f1(XS, YS)
        #     mcc = BASE.compute_mcc(XS, YS)

        #     if return_preds:
        #         _, y_pred = torch.max(XS, dim=1)
        #         return acc, loss, f1, mcc, y_pred, YS
        #     else:
        #         return acc, loss, f1, mcc

        # else:
        #     return XS
        
        if return_preds:
            _, y_pred = torch.max(XS, dim=1)
            return XS, y_pred
        else:
            return XS