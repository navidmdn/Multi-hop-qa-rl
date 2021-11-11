from graph_embeddings.models.embedding_model import EmbeddingModel
import torch


class ComplEx(EmbeddingModel):
    def __init__(self, data_loader, entity_dim, rel_dim, loss_type,
                 device, do_batch_norm, **kwargs):
        super(ComplEx, self).__init__(
            data_loader, entity_dim, rel_dim, loss_type,
            device, do_batch_norm, **kwargs
        )

        self.multiplier = 2
        self.entity_dim = entity_dim * self.multiplier
        self.bn0 = torch.nn.BatchNorm1d(self.multiplier)
        self.bn1 = torch.nn.BatchNorm1d(self.multiplier)
        self.bn2 = torch.nn.BatchNorm1d(self.multiplier)

    def calculate_score(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        if self.do_batch_norm:
            head = self.bn0(head)
        head = self.input_dropout(head)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        relation = self.hidden_dropout1(relation)
        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.E.weight, 2, dim=1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        if self.do_batch_norm:
            score = self.bn2(score)
        score = self.hidden_dropout2(score)
        score = score.permute(1, 0, 2)
        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1, 0)) + torch.mm(im_score, im_tail.transpose(1, 0))
        return score
