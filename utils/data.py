import torch
import pickle
import scipy.io
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from torch.utils.data import Dataset, DataLoader, TensorDataset

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Data:
    def __init__(self, file_path, num_neighbors):
        self.file_path = file_path
        if file_path == 'ng20text.tfidf.mat':
            # use ng20 with text to demonstrate case study
            self.load_datasets_ng20text()
        else:
            self.load_datasets()
        self.GetTopK_UsingCosineSim(TopK=num_neighbors, queryBatchSize=500, docBatchSize=100, useTest = False)
    
    def load_datasets(self):
        raise NotImplementedError

    def load_datasets_ng20text(self):
        raise NotImplementedError

    def GetTopK_UsingCosineSim(self, TopK, queryBatchSize, docBatchSize, useTest = False):
        raise NotImplementedError

    def get_loaders(self, num_trees, alpha, batch_size, num_workers, shuffle_train=False,
                    get_test=True):
        raise NotImplementedError


class LabeledDocuments(Data):
    def __init__(self, file_path, num_neighbors):
        super().__init__(file_path=file_path, num_neighbors=num_neighbors)

    def load_datasets(self):
        dataset = scipy.io.loadmat(self.file_path)

        # (num documents) x (vocab size) tensors containing tf-idf values
        self.X_train = torch.from_numpy(dataset['train'].toarray()).float()
        self.X_val = torch.from_numpy(dataset['cv'].toarray()).float()
        self.X_test = torch.from_numpy(dataset['test'].toarray()).float()

        # (num documents) x (num labels) tensors containing {0,1}
        self.Y_train = torch.from_numpy(dataset['gnd_train']).float()
        self.Y_val = torch.from_numpy(dataset['gnd_cv']).float()
        self.Y_test = torch.from_numpy(dataset['gnd_test']).float()

        self.vocab_size = self.X_train.size(1)
        self.num_labels = self.Y_train.size(1)
    
    def load_datasets_ng20text(self):
        dataset = gen_ng20_with_text()

        # (num documents) x (vocab size) tensors containing tf-idf values
        self.X_train = torch.from_numpy(dataset['train'].toarray()).float()
        self.X_val = torch.from_numpy(dataset['cv'].toarray()).float()
        self.X_test = torch.from_numpy(dataset['test'].toarray()).float()

        # (num documents) x (num labels) tensors containing {0,1}
        self.Y_train = torch.from_numpy(dataset['gnd_train']).float()
        self.Y_val = torch.from_numpy(dataset['gnd_cv']).float()
        self.Y_test = torch.from_numpy(dataset['gnd_test']).float()

        self.vocab_size = self.X_train.size(1)
        self.num_labels = self.Y_train.size(1)

        self.category = dataset['category']
        self.tokens = dataset['tokens']
        self.text_train = dataset['text_train']
        self.text_val = dataset['text_cv']
        self.text_test = dataset['text_test']

        # print(self.text_train[0], file=open(1, 'w', encoding='utf-8', closefd=False))
    
    def GetTopK_UsingCosineSim(self, TopK, queryBatchSize, docBatchSize, useTest = False):
        documents = deepcopy(self.X_train).type(FloatTensor)
        queries = deepcopy(self.X_test).type(FloatTensor) if useTest else deepcopy(self.X_train).type(FloatTensor)
        Y_documents = deepcopy(self.Y_train).type(FloatTensor)
        Y_queries = deepcopy(self.Y_test).type(FloatTensor) if useTest else deepcopy(self.Y_train).type(FloatTensor)

        # normalize 
        documents = documents / torch.norm(documents, p=2, dim=-1, keepdim=True)
        queries = queries / torch.norm(queries, p=2, dim=-1, keepdim=True)

        # compute cosine similarity
        cos_sim_scores = torch.mm(queries, documents.T)

        scores, indices = torch.topk(cos_sim_scores, TopK+1, dim=1, largest=True)
        self.topK_scores = scores[:, 1: ]
        self.topK_indices = indices[:, 1: ]

        # test 
        if useTest:
            print("test Top100 accuracy: {:.4f}".format(torch.mean((torch.sum(Y_queries.unsqueeze(1).repeat(1, TopK, 1) * Y_documents[self.topK_indices], dim = -1 ) > 0).type(FloatTensor)).item()))
            exit()
        else:
            print("graph (K={:d}) accuracy: {:.4f}".format(TopK, torch.mean((torch.sum(Y_queries.unsqueeze(1).repeat(1, TopK, 1) * Y_documents[self.topK_indices], dim = -1 ) > 0).type(FloatTensor)).item()))
        
        del documents, queries, Y_documents, Y_queries

    def get_loaders(self, num_trees, alpha, batch_size, num_workers, shuffle_train=False,
                    get_test=True):
        self.edges = self.get_spanning_trees(num_trees, alpha)
        self.num_nodes = self.X_train.shape[0]
        self.num_edges = self.edges.shape[0]

        train_dataset = TrainDataset(self.X_train, self.Y_train, self.edges)
        database_dataset = TestDataset(self.X_train, self.Y_train)
        val_dataset = TestDataset(self.X_val, self.Y_val)
        test_dataset = TestDataset(self.X_test, self.Y_test)

        # DataLoader
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                    shuffle=shuffle_train, num_workers=num_workers)
        database_loader = DataLoader(dataset=database_dataset, batch_size=512,
                                    shuffle=False, num_workers=num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=512,
                                    shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=512,
                                    shuffle=False, num_workers=num_workers) if get_test else None
        return train_loader, database_loader, val_loader, test_loader, 

    def get_spanning_trees(self, num_trees, alpha):
        edges = self.topK_indices.cpu().data.numpy()
        edges_scores = torch.softmax(FloatTensor(self.topK_scores) / alpha, dim=-1).cpu().data.numpy()

        N = edges.shape[0]
        w_m = {}
        for _ in range(num_trees):
            visited = np.array([False for i in range(N)])
            while False in visited:                                      
                init_node = np.random.choice(np.where(visited == False)[0], 1)[0]
                visited[init_node] = True
                queue = [init_node]
                while len(queue) > 0:
                    now = queue[0]
                    visited[now] = True
                    edge_idx = np.where(visited[edges[now]] == False)[0]
                    if len(edge_idx) == 0:
                        queue.pop(-1)
                        break
                    next = np.random.choice(edges[now][edge_idx], 1, p=edges_scores[now][edge_idx] / np.sum(edges_scores[now][edge_idx]))[0]
                    visited[next] = True
                    queue.append(next)
                    if (now * N + next) not in w_m:
                        w_m[now * N + next] = 1
                    else:
                        w_m[now * N + next] += 1
        
        edges = [[key // N, key % N, val / num_trees] for key, val in w_m.items()]
        np.random.shuffle(edges)
        return np.array(edges)

class TrainDataset(Dataset):
    def __init__(self, data, labels, edges):
        self.data = data
        self.labels = labels
        self.edges = edges
        
        self.edge_idx = 0

    def __getitem__(self, index):
        if self.edge_idx >= len(self.edges):
            self.edge_idx = 0
        text = self.data[index]
        labels = self.labels[index]
        edge1 = self.data[int(self.edges[self.edge_idx][0])]
        edge2 = self.data[int(self.edges[self.edge_idx][1])]
        weight = self.edges[self.edge_idx][2]
        self.edge_idx += 1
        return text, labels, edge1, edge2, weight
    
    def __len__(self):
        return len(self.data)

class TestDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)

def gen_ng20_with_text():
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    newsgroups_train = fetch_20newsgroups(subset='train', data_home='data')
    newsgroups_test = fetch_20newsgroups(subset='test', data_home='data')

    label_names = newsgroups_train.target_names
    train_data = newsgroups_train.data
    train_label = newsgroups_train.target
    test_data = newsgroups_test.data
    test_label = newsgroups_test.target
    all_data = train_data+test_data


    vectorizer = TfidfVectorizer(max_features = 10000)
    X = vectorizer.fit_transform(all_data)
    tokens = vectorizer.get_feature_names()

    train = X[:len(train_data)]
    test_all = X[len(train_data): ]

    gnd_train = np.zeros([len(train_data), 20])
    gnd_train[range(len(train_data)), train_label] = 1
    gnd_test_all = np.zeros([len(test_data), 20])
    gnd_test_all[range(len(test_data)), test_label] = 1

    tokens = tokens

    train = train
    gnd_train = gnd_train
    text_train = train_data

    test = test_all[: len(test_data) // 2]
    gnd_test = gnd_test_all[: len(test_data) // 2]
    text_test = test_data[: len(test_data) // 2]

    cv = test_all[len(test_data) // 2:]
    gnd_cv = gnd_test_all[len(test_data) // 2:]
    text_cv = test_data[len(test_data) // 2:]

    return {'tokens': tokens, 'category': label_names,
            'train': train, 'gnd_train': gnd_train, 'text_train': text_train,
            'test': test, 'gnd_test': gnd_test, 'text_test': text_test,
            'cv': cv, 'gnd_cv': gnd_cv, 'text_cv':text_cv}