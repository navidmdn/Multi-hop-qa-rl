from graph_embeddings.models.factory import EmbeddingModelFactory
from graph_embeddings.models.embedding_model import EmbeddingModel
from graph_embeddings.data_loader import DataLoader
from typing import Dict
import torch
import numpy as np
import os
import re


def generate_entity_embeddings(kge_model: EmbeddingModel, entities_dict: Dict, batch_size=5000):
    embeddings = {}
    entity_idx_list = list(entities_dict.values())
    kge_model.eval()

    i = 0
    while i < len(entity_idx_list):
        batch = entity_idx_list[i:i + batch_size]
        with torch.no_grad():
            emb_list = kge_model.E(torch.Tensor(batch).long()).cpu().numpy()

        for idx, emb in zip(batch, emb_list):
            embeddings[idx] = emb

        i += batch_size

    return embeddings


def generate_relation_embeddings(kge_model: EmbeddingModel, relations_dict: Dict):
    embeddings = {}
    rel_idx_list = list(relations_dict.values())
    kge_model.eval()

    with torch.no_grad():
        emb_list = kge_model.E(torch.Tensor(rel_idx_list).long()).cpu().numpy()

    for idx, emb in zip(rel_idx_list, emb_list):
        embeddings[idx] = emb

    return embeddings


def load_kge_model(dataset_name, model_name, ent_vec_dim, rel_vec_dim, loss_type, device, path, do_batch_norm=True,
                   reverse_rel=True, **kwargs):

    torch.backends.cudnn.deterministic = True
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        print("operating on gpu")
        torch.cuda.manual_seed_all(seed)
    else:
        print("operating on cpu")

    data_loader = DataLoader(dataset=dataset_name, reverse_rel=reverse_rel)
    embedding_generator = EmbeddingModelFactory(model_name).create(
            data_loader, ent_vec_dim, rel_vec_dim, loss_type, device, do_batch_norm, **kwargs
    )

    checkpoint = torch.load(os.path.join(data_loader.base_data_dir, path), map_location=torch.device(device))
    embedding_generator.load_state_dict(checkpoint)

    return embedding_generator


def extract_question_entity_target(raw_questions):
    all_questions = []
    all_entities = []
    all_targets = []

    for raw_q in raw_questions:
        question, targets = raw_q.split('\t')
        print(question, targets)
        entity = re.findall('\[.*?\]', question)[0] \
            .replace('[', '') \
            .replace(']', '')

        # todo: should I replace entity with some special token?
        question = question.replace(']', '').replace('[', '')
        targets = targets.strip().split('|')
        all_questions.append(question)
        all_targets.append(targets)
        all_entities.append(entity)

    return all_questions, all_entities, all_targets



