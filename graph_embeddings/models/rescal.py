from graph_embeddings.models.embedding_model import EmbeddingModel
import torch


class Rescal(EmbeddingModel):
    def __init__(self, data_loader, entity_dim, rel_dim, loss_type,
                 device, do_batch_norm, **kwargs):
        super(Rescal, self).__init__(
            data_loader, entity_dim, rel_dim, loss_type,
            device, do_batch_norm, **kwargs
        )

        self.multiplier = 1
        self.entity_dim = entity_dim * self.multiplier
        self.bn0 = torch.nn.BatchNorm1d(entity_dim * self.multiplier)
        self.bn1 = torch.nn.BatchNorm1d(entity_dim * self.multiplier)
        self.bn2 = torch.nn.BatchNorm1d(entity_dim * self.multiplier)

    def create_relation_embeddings(self):
        return torch.nn.Embedding(len(self.data_loader.relations), self.entity_dim * self.entity_dim, padding_idx=0)

    def calculate_score(self, head, relation):
        if self.do_batch_norm:
            head = self.bn0(head)
        head = self.input_dropout(head)
        head = head.view(-1, 1, self.entity_dim)
        relation = relation.view(-1, self.entity_dim, self.entity_dim)
        relation = self.hidden_dropout1(relation)
        x = torch.bmm(head, relation)
        x = x.view(-1, self.entity_dim)
        if self.do_batch_norm:
            x = self.bn2(x)
        x = self.hidden_dropout2(x)
        s = torch.mm(x, self.E.weight.transpose(1, 0))
        return s

