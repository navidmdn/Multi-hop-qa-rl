import os


class DataLoader:

    base_data_dir = '../data'

    def __init__(self, dataset="MetaQA", reverse_rel=False):
        """

        :param dataset: name of the dataset to load
        :param reverse_rel: if true, makes a reverse relation from original relations
        """

        self.entity_idxs = self.relation_idxs = None
        self.data_dir = os.path.join(self.base_data_dir, dataset)
        self.train_triples = self.load_triples("train", reverse_rel=reverse_rel)[:1000]
        self.valid_triples = self.load_triples("valid", reverse_rel=reverse_rel)[:1000]
        self.test_triples = self.load_triples("test", reverse_rel=reverse_rel)[:1000]
        self.data = self.train_triples + self.valid_triples + self.test_triples
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_triples)
        self.valid_relations = self.get_relations(self.valid_triples)
        self.test_relations = self.get_relations(self.test_triples)
        self.relations = list(set(self.train_relations + self.test_relations + self.valid_relations))
        self.store_triples()

    def load_triples(self, data_type="train", reverse_rel=False):

        with open(os.path.join(self.data_dir, f"{data_type}.txt"), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split('\t') for i in data]
            if reverse_rel:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]

        return data

    def store_triples(self):
        self.entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
        self.relation_idxs = {self.relations[i]: i for i in range(len(self.relations))}

        with open(os.path.join(self.data_dir, 'entities.dict'), 'w') as f:
            for key, value in self.entity_idxs.items():
                f.write(key + '\t' + str(value) + '\n')

        with open(os.path.join(self.data_dir, 'relations.dict'), 'w') as f:
            for key, value in self.relation_idxs.items():
                f.write(key + '\t' + str(value) + '\n')


    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities
