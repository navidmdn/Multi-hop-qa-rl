from graph_embeddings.models.embedding_model import EmbeddingModel
import torch
import numpy as np


class TuckER(EmbeddingModel):
    def __init__(self, data_loader, entity_dim, rel_dim, loss_type,
                 device, do_batch_norm, **kwargs):
        super(TuckER, self).__init__(
            data_loader, entity_dim, rel_dim, loss_type,
            device, do_batch_norm, **kwargs
        )

        self.multiplier = 1
        self.entity_dim = entity_dim * self.multiplier
        self.bn0 = torch.nn.BatchNorm1d(entity_dim * self.multiplier)
        self.bn1 = torch.nn.BatchNorm1d(entity_dim * self.multiplier)
        self.bn2 = torch.nn.BatchNorm1d(entity_dim * self.multiplier)

        self.W = torch.nn.Parameter(
            torch.tensor(np.random.uniform(-1, 1, (rel_dim, entity_dim, entity_dim)),
            dtype=torch.float, device=self.device, requires_grad=True)
        )

    def create_relation_embeddings(self):
        return torch.nn.Embedding(len(self.data_loader.relations), self.rel_dim, padding_idx=0)

    def calculate_score(self, head, relation):
        if self.do_batch_norm:
            head = self.bn0(head)
        ent_embedding_size = head.size(1)
        head = self.input_dropout(head)
        head = head.view(-1, 1, ent_embedding_size)

        W_mat = torch.mm(relation, self.W.view(relation.size(1), -1))
        W_mat = W_mat.view(-1, ent_embedding_size, ent_embedding_size)
        W_mat = self.hidden_dropout1(W_mat)

        s = torch.bmm(head, W_mat)
        s = s.view(-1, ent_embedding_size)
        s = self.bn2(s)
        s = self.hidden_dropout2(s)
        s = torch.mm(s, self.E.weight.transpose(1, 0))
        return s
