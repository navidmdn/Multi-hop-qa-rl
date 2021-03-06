import argparse
import os
import time
from collections import defaultdict
from typing import List, Dict

import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from graph_embeddings.data_loader import DataLoader
from graph_embeddings.models.factory import EmbeddingModelFactory
from torch.utils.tensorboard import SummaryWriter

class EmbeddingGenerator:

    def __init__(self, dataset, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200,
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False,
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0., valid_steps=1, loss_type='BCE', do_batch_norm=1,
                 model='TuckER', l3_reg=0.0, load_from='', output_model_name='default'):
        self.dataset = dataset

        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.valid_steps = valid_steps
        self.model_name = model
        self.l3_reg = l3_reg
        self.loss_type = loss_type
        self.load_from = load_from
        self.output_model_name = output_model_name

        self.tb_logger = SummaryWriter(
            comment=f"{self.dataset}-{self.model_name}_l3{self.l3_reg}_lr{self.learning_rate}_lrdecay{self.decay_rate}"
                    f"_batchsize{self.batch_size}_entdim{self.ent_vec_dim}_reldim{self.rel_vec_dim}_losstype({self.loss_type})"
        )

        if do_batch_norm == 1:
            do_batch_norm = True
        else:
            do_batch_norm = False
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2, "l3_reg": l3_reg}

        # TODO: configurable reverse rel
        self.data_loader = DataLoader(dataset=self.dataset, reverse_rel=True)
        device = 'cpu' if not self.cuda else 'cuda'
        self.model = EmbeddingModelFactory(self.model_name).create(
            self.data_loader, self.ent_vec_dim, rel_vec_dim, self.loss_type, device, do_batch_norm, **self.kwargs
        )

    def get_triple_idxs(self, triples: List) -> List:
        entity_idxs = self.data_loader.entity_idxs
        rel_idxs = self.data_loader.relation_idxs

        triple_idxs = [(entity_idxs[triple[0]], rel_idxs[triple[1]], entity_idxs[triple[2]])
                       for triple in triples]
        return triple_idxs

    @staticmethod
    def get_er_vocab(triples: List) -> Dict:
        er_vocab = defaultdict(list)
        for triple in triples:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = torch.zeros([len(batch), len(self.data_loader.entities)], dtype=torch.float32)
        if self.cuda:
            targets = targets.cuda()
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        return np.array(batch), targets

    def evaluate(self, model, data):
        model.eval()
        hits = [[] for _ in range(10)]
        ranks = []

        #TODO: its being called every time
        test_data_idxs = self.get_triple_idxs(data)
        er_vocab = self.get_er_vocab(test_data_idxs)

        print("Number of data points: %d" % len(test_data_idxs))
        # TODO: reuse get_batch function
        for i in tqdm(range(0, len(test_data_idxs), self.batch_size)):
            data_batch = np.array(test_data_idxs[i: i + self.batch_size])
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward(e1_idx, r_idx)

            # following lines commented means RAW evaluation (not filtered)
            # todo: whats this filtering?
            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j, e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                # kth rank of prediction matches target -> rank
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        hitat10 = np.mean(hits[9])
        hitat3 = np.mean(hits[2])
        hitat1 = np.mean(hits[0])
        meanrank = np.mean(ranks)

        mrr = np.mean(1. / np.array(ranks))
        print('Hits @10: {0}'.format(hitat10))
        print('Hits @3: {0}'.format(hitat3))
        print('Hits @1: {0}'.format(hitat1))
        print('Mean rank: {0}'.format(meanrank))
        print('Mean reciprocal rank: {0}'.format(mrr))

        return [mrr, meanrank, hitat10, hitat3, hitat1]

    def write_embedding_files(self, model):
        model_folder = os.path.join(self.data_loader.base_data_dir, f"{self.model_name}_{self.dataset}")
        torch.save(model.state_dict(), model_folder)

    def validation_step(self, model, best_valid, step):
        model.eval()
        with torch.no_grad():
            start_test = time.time()

            print("Train:")
            train = self.evaluate(model, self.data_loader.train_triples)
            print("Validation:")
            valid = self.evaluate(model, self.data_loader.valid_triples)
            print("Test:")
            test = self.evaluate(model, self.data_loader.test_triples)

            self.tb_logger.add_scalars('Score/mrr', {
                'train': train[0],
                'valid': valid[0],
                'test': test[0]
            }, step)

            self.tb_logger.add_scalars('Score/meanrank', {
                'train': train[1],
                'valid': valid[1],
                'test': test[1]
            }, step)

            self.tb_logger.add_scalars('Score/hit10', {
                'train': train[2],
                'valid': valid[2],
                'test': test[2]
            }, step)

            self.tb_logger.add_scalars('Score/hit3', {
                'train': train[3],
                'valid': valid[3],
                'test': test[3]
            }, step)

            self.tb_logger.add_scalars('Score/hit1', {
                'train': train[4],
                'valid': valid[4],
                'test': test[4]
            }, step)

            valid_mrr = valid[0]
            if valid_mrr >= best_valid[0]:
                best_valid = valid
                best_test = test
                print('Validation MRR increased.')
                print('Saving model...')
                self.write_embedding_files(model)
                print('Model saved!')

            print('Best valid:', best_valid)
            print('Best Test:', best_test)
            print('Dataset:', self.dataset)
            print('Model:', self.model_name)

            print(time.time() - start_test)
            print(
                'Learning rate %f | Decay %f | Dim %d | Input drop %f | Hidden drop 2 %f | LS %f | Batch size %d | Loss type %s | L3 reg %f' %
                (self.learning_rate, self.decay_rate, self.ent_vec_dim, self.kwargs["input_dropout"],
                 self.kwargs["hidden_dropout2"], self.label_smoothing, self.batch_size,
                 self.loss_type, self.l3_reg))

    def train_step(self, er_vocab_pairs, er_vocab, opt, model) -> List:
        losses = []
        for j in tqdm(range(0, len(er_vocab_pairs), self.batch_size)):
            data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
            opt.zero_grad()
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
            predictions = model.forward(e1_idx, r_idx)
            if self.label_smoothing:
                targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))
            loss = model.loss(predictions, targets)
            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu().numpy())
        return losses

    def train_and_eval(self):
        torch.set_num_threads(2)
        best_valid = [0, 0, 0, 0, 0]

        train_triple_idxs = self.get_triple_idxs(self.data_loader.train_triples)
        print("Number of training data points: %d" % len(train_triple_idxs))
        print('Entities: %d' % len(self.data_loader.entity_idxs))
        print('Relations: %d' % len(self.data_loader.relation_idxs))

        model = self.model
        model.init()

        if self.load_from != '':
            fname = self.load_from
            checkpoint = torch.load(os.path.join(self.data_loader.base_data_dir, fname))
            model.load_state_dict(checkpoint)
        if self.cuda:
            model.cuda()

        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_triple_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")

        for it in range(1, self.num_iterations + 1):
            print(f"iteration: {it}")
            print(scheduler.get_last_lr())
            self.tb_logger.add_scalar('training/lr', scheduler.get_last_lr()[0], it)

            start_train = time.time()
            model.train()
            np.random.shuffle(er_vocab_pairs)

            losses = self.train_step(er_vocab_pairs, er_vocab, opt, model)

            if self.decay_rate:
                scheduler.step()

            if it % self.valid_steps == 0:
                self.tb_logger.add_scalar('training/loss', np.mean(losses), it)
                print(f'Epoch:{it} Epoch time:{time.time() - start_train}, Loss:{np.mean(losses)}')
                self.validation_step(model, best_valid, it)

        self.tb_logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MetaQA", nargs="?",
                        help="Which dataset to use")
    parser.add_argument("--num_iterations", type=int, default=100, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                        help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                        help="Learning rate.")
    parser.add_argument("--model", type=str, default='TuckER', nargs="?",
                        help="Model.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                        help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                        help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                        help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=False, nargs="?",
                        help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                        help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                        help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                        help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                        help="Amount of label smoothing.")
    parser.add_argument("--valid_steps", type=int, default=1, nargs="?",
                        help="Epochs before u validate")
    parser.add_argument("--loss_type", type=str, default='CE', nargs="?",
                        help="Loss type")
    parser.add_argument("--do_batch_norm", type=int, default=1, nargs="?",
                        help="Do batch norm or not (0, 1)")
    parser.add_argument("--l3_reg", type=float, default=0.0, nargs="?",
                        help="l3 reg hyperparameter")
    parser.add_argument("--load_from", type=str, default='', nargs="?",
                        help="load from state dict")
    parser.add_argument("--output_model_name", type=str, default='default', nargs="?",
                        help="name of the saved model")

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available and args.cuda:
        print("operating on gpu")
        torch.cuda.manual_seed_all(seed)
    else:
        print("operating on cpu")

    embedding_generator = EmbeddingGenerator(
        dataset=args.dataset,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
        input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1,
        hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing,
        valid_steps=args.valid_steps, loss_type=args.loss_type, do_batch_norm=args.do_batch_norm,
        model=args.model, l3_reg=args.l3_reg, load_from=args.load_from, output_model_name=args.output_model_name
    )
    embedding_generator.train_and_eval()
