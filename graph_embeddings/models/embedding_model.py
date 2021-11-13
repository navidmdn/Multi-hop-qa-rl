import torch
from torch.nn.init import xavier_normal_
import torch.nn.functional as F

from abc import abstractmethod


class EmbeddingModel(torch.nn.Module):
    def __init__(self, data_loader, entity_dim, rel_dim,
                 loss_type, device='cpu', do_batch_norm=True, **kwargs):
        super(EmbeddingModel, self).__init__()

        self.loss_type = loss_type
        self.device = device
        self.do_batch_norm = do_batch_norm
        self.data_loader = data_loader
        self.entity_dim = entity_dim
        self.rel_dim = rel_dim

        if self.loss_type == 'BCE':
            self.loss = self.bce_loss
            self.bce_loss_loss = torch.nn.BCELoss()
        elif self.loss_type == 'CE':
            self.loss = self.ce_loss
        elif self.loss_type == 'BCELOG':
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f'Incorrect loss specified: {self.loss_type}')

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.l3_reg = kwargs["l3_reg"]

    def freeze_entity_embeddings(self):
        self.E.weight.requires_grad = False
        print('Entity embeddings are frozen')

    def create_relation_embeddings(self):
        return torch.nn.Embedding(len(self.data_loader.relations), self.entity_dim * self.multiplier, padding_idx=0)

    def create_entity_embeddings(self):
        return torch.nn.Embedding(len(self.data_loader.entities), self.entity_dim * self.multiplier, padding_idx=0)

    def ce_loss(self, pred, true):
        pred = F.log_softmax(pred, dim=-1)
        true = true / true.size(-1)
        loss = -torch.sum(pred * true)
        return loss

    def bce_loss(self, pred, true):
        loss = self.bce_loss_loss(pred, true)
        # l3 regularization
        if self.l3_reg:
            norm = torch.norm(self.E.weight.data, p=3, dim=-1)
            loss += self.l3_reg * torch.sum(norm)
        return loss

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    @abstractmethod
    def calculate_score(self, head, relation):
        pass

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        h = e1
        r = self.R(r_idx)
        ans = self.calculate_score(h, r)
        pred = torch.sigmoid(ans)
        return pred
