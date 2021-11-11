from graph_embeddings.models.simple import SimplE
from graph_embeddings.models.complex import ComplEx
from graph_embeddings.models.rescal import Rescal
from graph_embeddings.models.tucker import TuckER
from graph_embeddings.models.dist_mult import DistMult

class EmbeddingModelFactory:
    def __init__(self, model_name):
        self.model_name = model_name

    def create(self, data_loader, entity_dim, rel_dim, loss_type, device, do_batch_norm, **kwargs):
        if self.model_name == 'DistMult':
            print("building distmult model for embedding generation")
            embedding_model = DistMult(data_loader, entity_dim, rel_dim, loss_type, device, do_batch_norm, **kwargs)
        elif self.model_name == 'SimplE':
            print("building simple model for embedding generation")
            embedding_model = SimplE(data_loader, entity_dim, rel_dim, loss_type, device, do_batch_norm, **kwargs)
        elif self.model_name == 'ComplEx':
            print("building complex model for embedding generation")
            embedding_model = ComplEx(data_loader, entity_dim, rel_dim, loss_type, device, do_batch_norm, **kwargs)
        elif self.model_name == 'RESCAL':
            print("building rescal model for embedding generation")
            embedding_model = Rescal(data_loader, entity_dim, rel_dim, loss_type, device, do_batch_norm, **kwargs)
        elif self.model_name == 'TuckER':
            print("building tucker model for embedding generation")
            embedding_model = TuckER(data_loader, entity_dim, rel_dim, loss_type, device, do_batch_norm, **kwargs)
        else:
            raise NotImplementedError("Wrong model name!")

        return embedding_model
