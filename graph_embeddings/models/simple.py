from graph_embeddings.models.embedding_model import EmbeddingModel
import torch


class SimplE(EmbeddingModel):
    def __init__(self, data_loader, entity_dim, rel_dim, loss_type,
                 device, do_batch_norm, **kwargs):
        super(SimplE, self).__init__(
            data_loader, entity_dim, rel_dim, loss_type,
            device, do_batch_norm, **kwargs
        )

        self.multiplier = 1
        self.entity_dim = entity_dim * self.multiplier
        self.bn0 = torch.nn.BatchNorm1d(entity_dim * self.multiplier)
        self.bn1 = torch.nn.BatchNorm1d(entity_dim * self.multiplier)
        self.bn2 = torch.nn.BatchNorm1d(entity_dim * self.multiplier)

    def calculate_score(self, head, relation):
        if self.do_batch_norm:
            head = self.bn0(head)
        head = self.input_dropout(head)
        relation = self.hidden_dropout1(relation)
        s = head * relation
        s_head, s_tail = torch.chunk(s, 2, dim=1)
        s = torch.cat([s_tail, s_head], dim=1)
        if self.do_batch_norm:
            s = self.bn2(s)
        s = self.hidden_dropout2(s)
        s = torch.mm(s, self.E.weight.transpose(1, 0))
        s = 0.5 * s
        return s
