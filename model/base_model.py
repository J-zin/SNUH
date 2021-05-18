import math
import torch
import random
import argparse
import datetime
import numpy as np
import torch.nn as nn
from copy import deepcopy
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patheffects as pe
from collections import OrderedDict
from timeit import default_timer as timer

from utils.logger import Logger
from utils.data import LabeledDocuments
from utils.evaluation import compute_retrieval_precision, compute_retrieval_precision_median_threshold, compute_hamming_distance, cosine_distance_torch

class Base_Model(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.load_data()
    
    def load_data(self):
        self.data = LabeledDocuments(self.hparams.data_path, self.hparams.num_neighbors)
    
    def configure_optimizers(self):
        raise NotImplementedError

    def configure_gradient_clippers(self):
        raise NotImplementedError

    def encode_discrete(self, target_inputs):
        raise NotImplementedError

    def get_median_threshold_binary_code(self, device, is_val=True):
        raise NotImplementedError

    def run_training_sessions(self):
        logger = Logger(self.hparams.model_path + '.log', on=True)
        val_perfs = []
        best_val_perf = float('-inf')
        start = timer()
        random.seed(self.hparams.seed)  # For reproducible random runs

        for run_num in range(1, self.hparams.num_runs + 1):
            state_dict, val_perf = self.run_training_session(run_num, logger)
            val_perfs.append(val_perf)

            if val_perf > best_val_perf:
                best_val_perf = val_perf
                logger.log('----New best {:8.2f}, saving'.format(val_perf))
                torch.save({'hparams': self.hparams,
                            'state_dict': state_dict}, self.hparams.model_path)
        
        logger.log('Time: %s' % str(timedelta(seconds=round(timer() - start))))
        self.load()
        if self.hparams.num_runs > 1:
            logger.log_perfs(val_perfs)
            logger.log('best hparams: ' + self.flag_hparams())
        
        val_perf, test_perf = self.run_test()
        logger.log('Val:  {:8.2f}'.format(val_perf))
        logger.log('Test: {:8.2f}'.format(test_perf))

    def run_training_session(self, run_num, logger):
        self.train()

        # Scramble hyperparameters if number of runs is greater than 1.
        if self.hparams.num_runs > 1:
            logger.log('RANDOM RUN: %d/%d' % (run_num, self.hparams.num_runs))
            for hparam, values in self.get_hparams_grid().items():
                assert hasattr(self.hparams, hparam)
                self.hparams.__dict__[hparam] = random.choice(values)
        
        np.random.seed(self.hparams.seed)
        random.seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)

        self.define_parameters()
        # logger.log(str(self))
        logger.log('%d params' % sum([p.numel() for p in self.parameters()]))
        logger.log('hparams: %s' % self.flag_hparams())
        
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        self.to(device)
        self.tau = self.tau.to(device)

        optimizer = self.configure_optimizers()
        gradient_clippers = self.configure_gradient_clippers()
        train_loader, database_loader, val_loader, _ = self.data.get_loaders(
            self.hparams.num_trees, self.hparams.alpha, self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=True, get_test=False)
        best_val_perf = float('-inf')
        best_state_dict = None
        forward_sum = {}
        num_steps = 0
        bad_epochs = 0
        kl_weight = 0.

        times = []
        try:
            for epoch in range(1, self.hparams.epochs + 1):
                starttime = datetime.datetime.now()
                for batch_num, batch in enumerate(train_loader):
                    optimizer.zero_grad()

                    X, edge1, edge2, weight = batch[0].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
                    if self.hparams.no_tfidf:
                        X, edge1, edge2 = X.sign(), edge1.sign(), edge2.sign()
                    
                    forward = self.forward(X, edge1, edge2, weight, self.data.num_nodes, self.data.num_edges)
                    for key in forward:
                        if key in forward_sum:
                            forward_sum[key] += forward[key]
                        else:
                            forward_sum[key] = forward[key]
                    num_steps += 1

                    if math.isnan(forward_sum['loss']):
                        logger.log('Stopping epoch because loss is NaN')
                        break
                        
                    forward['loss'].backward()

                    for params, clip in gradient_clippers:
                        nn.utils.clip_grad_norm_(params, clip)

                    optimizer.step()
                
                endtime = datetime.datetime.now()
                times.append(endtime - starttime)

                if math.isnan(forward_sum['loss']):
                    logger.log('Stopping training session because loss is NaN')
                    break

                val_perf = self.evaluate(database_loader, val_loader, device, is_val=True)
                logger.log('End of epoch {:3d}'.format(epoch), False)
                logger.log(' '.join([' | {:s} {:8.2f}'.format(
                    key, forward_sum[key] / num_steps)
                                     for key in forward_sum]), False)
                logger.log(' | val perf {:8.2f}'.format(val_perf), False)

                if val_perf > best_val_perf:
                    best_val_perf = val_perf
                    bad_epochs = 0
                    logger.log('\t\t*Best model so far, deep copying*')
                    best_state_dict = deepcopy(self.state_dict())
                else:
                    bad_epochs += 1
                    logger.log('\t\tBad epoch %d' % bad_epochs)

                if bad_epochs > self.hparams.num_bad_epochs:
                    break
        except KeyboardInterrupt:
            logger.log('-' * 89)
            logger.log('Exiting from training early')

        logger.log("time per training epoch: " + str(np.mean(times)))
        return best_state_dict, best_val_perf
    
    def evaluate(self, database_loader, eval_loader, device, is_val=True):
        self.eval()
        with torch.no_grad():
            # Retrieval prec of eval wrt labeled docs in train
            if self.hparams.median:
                train_b, test_b, train_y, test_y = self.get_median_threshold_binary_code(device, is_val)
                perf = compute_retrieval_precision_median_threshold(train_b, test_b, train_y, test_y, self.hparams.distance_metric, self.hparams.num_retrieve)
            else:
                perf = compute_retrieval_precision(database_loader, eval_loader,
                                                    device, self.encode_discrete,
                                                    self.hparams.distance_metric,
                                                    self.hparams.num_retrieve)
        self.train()
        return perf

    def load(self):
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        checkpoint = torch.load(self.hparams.model_path) if self.hparams.cuda \
                     else torch.load(self.hparams.model_path,
                                     map_location=torch.device('cpu'))
        if checkpoint['hparams'].cuda and not self.hparams.cuda:
            checkpoint['hparams'].cuda = False
        self.hparams = checkpoint['hparams']
        self.define_parameters()
        self.load_state_dict(checkpoint['state_dict'])
        self.to(device)
        self.tau = self.tau.to(device)

    def run_test(self):
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        _, database_loader, val_loader, test_loader = self.data.get_loaders(
            self.hparams.num_trees, self.hparams.alpha, self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=False, get_test=True)
        val_perf = self.evaluate(database_loader, val_loader, device, is_val=True)
        test_perf = self.evaluate(database_loader, test_loader, device, is_val=False)
        return val_perf, test_perf

    def hash_codes_visulization(self):
        self.eval()
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        _, database_loader, _, _ = self.data.get_loaders(
            self.hparams.num_trees, self.hparams.alpha, self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=False, get_test=True)
        
        def extract_data(loader):
            encoding_chunks = []
            label_chunks = []
            for (docs, labels) in loader:
                docs = docs.to(device)
                encoding_chunks.append(docs if self.encode_discrete is None else
                                    self.encode_discrete(docs))
                label_chunks.append(labels)
            encoding_mat = torch.cat(encoding_chunks, 0)
            label_mat = torch.cat(label_chunks, 0)
            label_lists = [[j.item() for j in label_mat[i].nonzero()] for i in
                        range(label_mat.size(0))]
            return encoding_mat, label_lists
        
        labels_ticks = ['atheism', 'graphics', 'ms-windows.misc', 'hardware', 'hardware', 'windows.x', 'forsale', 'autos', 'motorcycles', 
            'baseball', 'hockey', 'crypt', 'electronics', 'sci.med', 'space', 'christian', 'guns', 'mideast', 'politics.misc', 'religion.misc']
        hash_codes, labels = extract_data(database_loader)
        hash_codes, labels = hash_codes.cpu().detach().numpy(), np.array(labels).reshape(-1)

        # TSN
        mapper = TSNE(perplexity=30).fit_transform(hash_codes)

        plt.figure(figsize=(8.5, 8))
        plt.scatter(mapper[:,0], mapper[:,1], lw=0, s=20, c=labels.astype(np.int), cmap='Spectral')
        
        for i in range(len(labels_ticks)):
            # Position of each label.
            xtext, ytext = np.median(mapper[labels == i, :], axis=0)
            txt = plt.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        
        plt.axis("off")
        plt.gcf().tight_layout()
        plt.savefig('Ours_hash_codes_visulization_{:d}bits.pdf'.format(self.hparams.num_features), bbox_inches='tight', pad_inches=0.0)

    def flag_hparams(self):
        flags = '%s %s' % (self.hparams.model_path, self.hparams.data_path)
        for hparam in vars(self.hparams):
            val = getattr(self.hparams, hparam)
            if str(val) == 'False':
                continue
            elif str(val) == 'True':
                flags += ' --%s' % (hparam)
            elif str(hparam) in {'model_path', 'data_path', 'num_runs',
                                 'num_workers'}:
                continue
            else:
                flags += ' --%s %s' % (hparam, val)
        return flags

    @staticmethod
    def get_general_hparams_grid():
        grid = OrderedDict({
            'seed': list(range(100000)),
            'lr': [0.003, 0.001, 0.0005],
            'clip': [1, 5, 10],
            'batch_size': [32, 64, 128],
            'init': [0, 0.5, 0.1, 0.05, 0.01],
            })
        return grid

    @staticmethod
    def get_general_argparser():
        parser = argparse.ArgumentParser()

        parser.add_argument('model_path', type=str)
        parser.add_argument('data_path', type=str)
        parser.add_argument('--train', action='store_true',
                            help='train a model?')

        parser.add_argument('--num_features', type=int, default=64,
                            help='num discrete features [%(default)d]')
        parser.add_argument('--dim_hidden', type=int, default=500,
                            help='dimension of hidden state [%(default)d]')
        parser.add_argument('--num_layers', type=int, default=0,
                            help='num layers [%(default)d]')
        parser.add_argument('--num_neighbors', type=int, default=10,
                            help='num neighbors [%(default)d]')
        
        parser.add_argument('--batch_size', type=int, default=128,
                            help='batch size [%(default)d]')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='initial learning rate [%(default)g]')
        parser.add_argument('--init', type=float, default=0.05,
                            help='unif init range (default if 0) [%(default)g]')
        parser.add_argument('--clip', type=float, default=10,
                            help='gradient clipping [%(default)g]')
        parser.add_argument('--epochs', type=int, default=100,
                            help='max number of epochs [%(default)d]')
        parser.add_argument('--num_runs', type=int, default=1,
                            help='num random runs (not random if 1) '
                            '[%(default)d]')

        parser.add_argument('--num_retrieve', type=int, default=100,
                            help='num neighbors to retrieve [%(default)d]')
        parser.add_argument('--num_bad_epochs', type=int, default=6,
                            help='num indulged bad epochs [%(default)d]')
        parser.add_argument('--num_workers', type=int, default=0,
                            help='num dataloader workers [%(default)d]')
        parser.add_argument('--distance_metric', default='hamming',
                            choices=['hamming', 'cosine']),
        parser.add_argument('--no_tfidf', action='store_true',
                            help='raw bag-of-words as input instead of tf-idf?')
        parser.add_argument('--seed', type=int, default=50971,
                            help='random seed [%(default)d]')
        parser.add_argument('--cuda', action='store_true',
                            help='use CUDA?')
        parser.add_argument('--median', action='store_true',
                            help='median threshold (VDSH) ?')

        return parser