from os.path import join, abspath, dirname
import networkx as nx
import pickle


class DataLoader:

    base_data_dir = join(dirname(dirname(abspath(__file__))), 'data')

    def __init__(self, dataset="MetaQA", reverse_rel=False):
        """

        :param dataset: name of the dataset to load
        :param reverse_rel: if true, makes a reverse relation from original relations
        """

        self.entity_idxs = self.relation_idxs = None
        self.data_dir = join(self.base_data_dir, dataset)
        self.train_triples = self.load_triples("train", reverse_rel=reverse_rel)
        self.valid_triples = self.load_triples("valid", reverse_rel=reverse_rel)
        self.test_triples = self.load_triples("test", reverse_rel=reverse_rel)
        self.data = self.train_triples + self.valid_triples + self.test_triples

        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_triples)
        self.valid_relations = self.get_relations(self.valid_triples)
        self.test_relations = self.get_relations(self.test_triples)
        self.relations = list(set(self.train_relations + self.test_relations + self.valid_relations))
        self.store_triples()

    def load_triples(self, data_type="train", reverse_rel=False):

        with open(join(self.data_dir, f"{data_type}.txt"), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split('\t') for i in data]
            if reverse_rel:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]

        return data

    def store_triples(self):
        self.entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
        self.relation_idxs = {self.relations[i]: i for i in range(len(self.relations))}

        with open(join(self.data_dir, 'entities.dict'), 'w') as f:
            for key, value in self.entity_idxs.items():
                f.write(key + '\t' + str(value) + '\n')

        with open(join(self.data_dir, 'relations.dict'), 'w') as f:
            for key, value in self.relation_idxs.items():
                f.write(key + '\t' + str(value) + '\n')

    def load_entity_relations_vocab(self):
        entities = {}
        relations = {}

        with open(join(self.data_dir, 'entities.dict'), 'r') as f:
            entities_txt = f.read().strip().split('\n')
            for line in entities_txt:
                line = line.split('\t')
                entity, idx = line[0], line[1]
                entities[entity] = int(idx)

        with open(join(self.data_dir, 'relations.dict'), 'r') as f:
            relations_txt = f.read().strip().split('\n')
            for line in relations_txt:
                line = line.split('\t')
                rel, idx = line[0], line[1]
                relations[rel] = int(idx)

        return entities, relations

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities

    def build_graph(self, save_path=None):
        graph = nx.DiGraph()

        entities_voc, relations_voc = self.load_entity_relations_vocab()

        for triple in self.data:
            n1, n2 = entities_voc[triple[0]], entities_voc[triple[2]]
            e = relations_voc[triple[1]]

            graph.add_nodes_from([
                (n1, {"entity_text": triple[0]}),
                (n2, {"entity_text": triple[1]})
            ])

            graph.add_edge(n1, n2, relation_id=e, relation_text=triple[1])

        if save_path:
            with open(join(self.data_dir, 'graph.pickle'), 'wb') as f:
                pickle.dump(graph, f)

        return graph

