from graph_embeddings.models.embedding_model import EmbeddingModel


class DistMult(EmbeddingModel):
    def __init__(self, data_loader, entity_dim, rel_dim, loss_type,
                 device, do_batch_norm, **kwargs):
        super(DistMult, self).__init__(
            data_loader, entity_dim, rel_dim, loss_type,
            device, do_batch_norm, **kwargs
        )

        self.multiplier = 1
        self.entity_dim = entity_dim * self.multiplier



