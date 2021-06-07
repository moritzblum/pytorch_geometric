import math
from typing import List
import os
import json
import urllib.request

from torch import nn
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdflib import Graph, term
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.nn import GAE, RGCNConv


# manage dataset 
class GeneralRDFDataset(InMemoryDataset):

    def __init__(self, root='../data', name='rdf_test', transform=None,
                 pre_transform=None):
        """
        :param root: root directory containing the data
        :param name: rdf file with .nt suffix

        """

        if name in ['FB15k-237']:
            self.name = name
            super(GeneralRDFDataset, self).__init__(root, transform, pre_transform)
            self.data, self.slices = torch.load(self.processed_paths[0])
            self.data.num_relations = self.data.num_relations.tolist()  # convert manually, because not managed by PyTG


    def download(self):
        if self.name == 'FB15k-237':
            urllib.request.urlretrieve("https://raw.githubusercontent.com/MichSchli/RelationPrediction/master/data/FB-Toutanova/README.txt", f"{self.raw_dir}/README.txt")
            urllib.request.urlretrieve("https://raw.githubusercontent.com/MichSchli/RelationPrediction/master/data/FB-Toutanova/entities.dict", f"{self.raw_dir}/entities.dict")
            urllib.request.urlretrieve("https://raw.githubusercontent.com/MichSchli/RelationPrediction/master/data/FB-Toutanova/relations.dict", f"{self.raw_dir}/relations.dict")
            urllib.request.urlretrieve("https://raw.githubusercontent.com/MichSchli/RelationPrediction/master/data/FB-Toutanova/test.txt", f"{self.raw_dir}/test.txt")
            urllib.request.urlretrieve("https://raw.githubusercontent.com/MichSchli/RelationPrediction/master/data/FB-Toutanova/train.txt", f"{self.raw_dir}/train.txt")
            urllib.request.urlretrieve("https://raw.githubusercontent.com/MichSchli/RelationPrediction/master/data/FB-Toutanova/valid.txt", f"{self.raw_dir}/valid.txt")
        else:
            print(f'You must put your rdf in n-triple format into {self.raw_paths}.')

    def process(self):
        x, edge_index, edge_type = read_rgcn_data(self.raw_dir)
        data = Data(x=x, edge_index=torch.reshape(edge_index, (2, -1)), edge_type=edge_type)

        unique_nodes = torch.unique(edge_index)
        data.num_nodes = max(unique_nodes.size(0), torch.max(unique_nodes) + 1).item()

        data.num_relations = data.edge_type.max().item() + 1

        self.data, self.slices = self.collate([data])
        torch.save((self.data, self.slices), self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    @property
    def raw_file_names(self):
    	return ['entities.dict', 'relations.dict', 'test.txt', 'train.txt', 'valid.txt']

    @property
    def num_relations(self):
        return self.data.edge_type.max().item() + 1

    @property
    def num_nodes(self):
        unique_nodes = torch.unique(data.edge_index)
        return max(unique_nodes.size(0), torch.max(unique_nodes) + 1).item()
        
         
def read_rgcn_data(directory):

    # read dict
    entities_dict = {}
    for line in open(os.path.join(directory, 'entities.dict'), 'r+'):
        line = line.strip().split('\t')
        entities_dict[line[1]] = int(line[0])

    relations_dict = {}
    for line in open(os.path.join(directory, 'relations.dict'), 'r+'):
        line = line.strip().split('\t')
        relations_dict[line[1]] = int(line[0])

    subjects = []
    relations = []
    objects = []

    for line in open(os.path.join(directory, 'train.txt'), 'r+'):
        subject, relation, object = line.strip().split('\t')
        subjects.append(entities_dict[subject])
        relations.append(relations_dict[relation])
        objects.append(entities_dict[object])

    x = torch.tensor([])

    return x, torch.tensor([subjects, objects]), torch.tensor(relations)     
    
    
# link prediction model
class RGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(RGCNEncoder, self).__init__()
        self.linear_features = nn.Linear(in_channels, 500)
        self.conv1 = RGCNConv(500, out_channels, num_relations=num_relations, num_blocks=5)
        self.conv2 = RGCNConv(out_channels, out_channels, num_relations=num_relations, num_blocks=5)

    def forward(self, x, edge_index, edge_type):
        x = self.linear_features(x)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x

class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, relation_embedding_dim):
        super(DistMultDecoder, self).__init__()
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, relation_embedding_dim))
        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

    def forward(self, z, edge_index, edge_type):
        s = z[edge_index[0, :]]
        r = self.relation_embedding[edge_type]
        o = z[edge_index[1, :]]
        score = torch.sum(s * r * o, dim=1)
        return score
        
        
def negative_sampling(edge_index, edge_type, num_nodes, omega=1.0, corrupt_subject=True, corrupt_predicate=False, corrupt_object=True):

    negative_samples_count = int(omega * edge_type.size(0))
    permutation = np.random.choice(np.arange(edge_type.size(0)), negative_samples_count)
    s = edge_index[0][permutation]
    r = edge_type[permutation]
    o = edge_index[0][permutation]

    permutation_variants = np.random.choice(np.array([0, 1, 2])[[corrupt_subject, corrupt_predicate, corrupt_object]],
                                            negative_samples_count)

    random_nodes = np.random.choice(np.arange(num_nodes), negative_samples_count)
    # todo allow more edge types than contained in the true data
    random_labels = np.random.choice(np.arange(edge_type.max().item() + 1), negative_samples_count)

    # replace subjects with random node
    s_permutation_mask = permutation_variants == 0
    s_true_nodes = s.clone()
    s_true_nodes[s_permutation_mask] = 0
    s_random_nodes = torch.tensor(random_nodes)
    s_random_nodes[~s_permutation_mask] = 0
    s_permuted = s_true_nodes + s_random_nodes

    # replace edge labels with random label
    r_permutation_mask = permutation_variants == 1
    r_true_predicates = r.clone()
    r_true_predicates[r_permutation_mask] = 0
    r_random_labels = torch.tensor(random_labels)
    r_random_labels[~r_permutation_mask] = 0
    negative_sample_edge_type = r_true_predicates + r_random_labels

    # replace objects with random node
    o_permutation_mask = permutation_variants == 2
    o_true_nodes = o.clone()
    o_true_nodes[o_permutation_mask] = 0
    o_random_nodes = torch.tensor(random_nodes)
    o_random_nodes[~o_permutation_mask] = 0
    o_permuted = o_true_nodes + o_random_nodes

    # remove sampled triples which are contained in the real graph and therefore no negative samples
    overlap_to_true_triples = (s == s_permuted) * (r == negative_sample_edge_type) * (o == o_permuted)
    if negative_samples_count != 0:
        #print('generated negative samples which overlap with true triples and are now removed:',
        #    round(torch.count_nonzero(overlap_to_true_triples).item() / negative_samples_count, 2) * 100, '%')
        pass

    s_permuted = s_permuted[~overlap_to_true_triples]
    negative_sample_edge_type = negative_sample_edge_type[~overlap_to_true_triples]
    o_permuted = o_permuted[~overlap_to_true_triples]

    neg_samples_mask = torch.cat((torch.ones(edge_index.size(1), dtype=torch.bool),
                                       torch.zeros(negative_sample_edge_type.size(0), dtype=torch.bool)), 0)
    edge_index = torch.cat((edge_index, torch.cat((s_permuted, o_permuted), 0).reshape(2, -1)), 1)
    edge_type = torch.cat((edge_type, negative_sample_edge_type), 0)
    return edge_index, edge_type, neg_samples_mask
    
    
def train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1) -> Data:
    """
    Implements simplified train/val/test edges split similar to the PyTorch Geometric function, but involving edge_type.
    :param data:
    :param val_ratio:
    :param test_ratio:
    :return: Data object with train/val/test split into the variables: val_edge_index, test_edge_index,
    train_edge_index, val_edge_type, test_edge_type, train_edge_type
    """

    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index

    data.edge_index = None

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_edge_index = torch.stack([r, c], dim=0)

    # same for the edge attr
    attr = None
    if hasattr(data, 'edge_type') is not None:
        attr = data.edge_type
    if attr is not None:
        data.edge_type = None
        attr = attr[perm]
        data.val_edge_type = attr[:n_v]
        data.test_edge_type = attr[n_v:n_v + n_t]
        data.train_edge_type = attr[n_v + n_t:]

    return data
         
        
def sort_and_rank(score, target):
    _, indices = torch.sort(score, descending=True)
    rank = torch.reshape(torch.nonzero(indices == target), (-1, )).item() + 1
    return rank
    
    
def get_ranks(scores, true_edges) -> np.ndarray:
    """
    Computes the ranks for the edges marked as true edges (ones in true_edges) according to the given scores.
    """
    y_pred = np.array(scores)
    y_true = np.array(true_edges)

    if y_pred.size != y_true.size or y_pred.ndim != 1 or y_true.ndim != 1:
        raise ArithmeticError('input not valid (check size and shape)')

    idx = np.argsort(y_pred)[::-1]
    y_ord = y_true[idx]
    ranks = np.where(y_ord == 1)[0] + 1

    # true edges do not affect the rank: decrease each rank of true edges by the number of true edges ranked before
    ranks_cleared = ranks - np.arange(len(ranks))

    return ranks_cleared
    

def evaluate_relational_link_prediction(latent_node_variables, w, test_triplets, all_triplets) -> dict:
    """
    Evaluation according to Borders et al. "Translating Embeddings for Modeling Multi-relational Data" (2013).
    """
    with torch.no_grad():
        num_nodes = latent_node_variables.size(0)
        ranks_r = {r: [] for r in np.unique(all_triplets[:, 1])}  # stores ranks computed for each relation

        for triplet in tqdm(test_triplets):
            head, relation, tail = triplet

            # permute object
            # try all nodes as tails, but delete all true triplets with the same head and relation as the test triplet
            delete_triplet_index = torch.nonzero(torch.logical_and(all_triplets[:, 0] == head, all_triplets[:, 1] == relation))
            delete_entity_index = torch.flatten(all_triplets[delete_triplet_index, 2]).numpy()

            tails = torch.cat((torch.from_numpy(np.array(list(set(np.arange(num_nodes)) - set(delete_entity_index)))), tail.view(-1)))

            # head nods and relations are all the same
            heads = torch.zeros(tails.size(0)).fill_(head).type(torch.long)
            edge_types = torch.zeros(tails.size(0)).fill_(relation).type(torch.long)

            s = latent_node_variables[heads]
            r = w[edge_types]
            o = latent_node_variables[tails]
            scores = torch.sum(s * r * o, dim=1)
            scores = torch.sigmoid(scores)

            target = torch.tensor(len(tails) - 1)

            rank = sort_and_rank(scores, target)


            ranks_r[relation.item()].append(rank)

            # permute subject
            delete_triplet_index = torch.nonzero(torch.logical_and(all_triplets[:, 1] == relation, all_triplets[:, 2] == tail))
            delete_entity_index = torch.flatten(all_triplets[delete_triplet_index, 0]).numpy()

            heads = torch.cat((torch.from_numpy(np.array(list(set(np.arange(num_nodes)) - set(delete_entity_index)))), head.view(-1)))

            # tail nods and relations are all the same
            edge_types = torch.zeros(heads.size(0)).fill_(relation).type(torch.long)
            tails = torch.zeros(heads.size(0)).fill_(tail).type(torch.long)

            s = latent_node_variables[heads]
            r = w[edge_types]
            o = latent_node_variables[tails]
            scores = torch.sum(s * r * o, dim=1)
            scores = torch.sigmoid(scores)

            target = torch.tensor(len(tails) - 1)
            rank = sort_and_rank(scores, target)
            ranks_r[relation.item()].append(rank)

        # compute scores
        k = []
        for x in ranks_r.values(): k.extend(x)
        ranks = torch.tensor(k)
        scores = {'MRR': torch.mean(1.0 / ranks.float()).item(),
                  'H@1': torch.mean((ranks <= 1).float()).item(),
                  'H@3': torch.mean((ranks <= 3).float()).item(),
                  'H@10': torch.mean((ranks <= 10).float()).item()}

    return scores    
    
    
        
if __name__ == '__main__':
    mode = 'GAE-RGCN'
    epochs = 10000
    out_channels = 500
    reg_ratio = 1e-2
    batch_size = 1000

    dataset = GeneralRDFDataset('../data', 'FB15k-237')   

    data = dataset[0]
    if data.x is None or data.x.size(0) == 0:
        print('one hot encoding')
        data.x = torch.eye(data.num_nodes + 1).to_sparse()

    num_features = data.num_features
    data = train_test_split_edges(data)

    print('num nodes:', data.num_nodes)
    print('num_relations:', data.num_relations)

    model = None
    if mode == 'GAE-RGCN':
        model = GAE(RGCNEncoder(num_features, out_channels, num_relations=data.num_relations * 2),
                    DistMultDecoder(data.num_relations * 2, out_channels))
    else:
        KeyError(f'Mode "{mode}" does not exist. You can use: GAE-GCN, VGAE-GCN, ...')

    # auto-encoder training is independent from the used encoder model
    device = torch.device('cpu')  # 'cuda' if torch.cuda.is_available() else 'cpu')
    # torch.set_num_threads(4)

    model = model.to(device)
    x = data.x.to(device)

    train_edge_index_all = data.train_edge_index
    train_edge_types_all = data.train_edge_type

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in tqdm(range(epochs)):
        device = torch.device('cpu')  # 'cuda' if torch.cuda.is_available() else 'cpu')

        model.train()
        optimizer.zero_grad()

        # sample random batch of triples
        batch_perm = np.arange(train_edge_index_all.size(1))
        np.random.shuffle(batch_perm)
        if batch_size:
            batch_perm = torch.tensor(batch_perm[:batch_size])
        else:
            batch_perm = torch.tensor(batch_perm)
        train_edge_index_batch = torch.index_select(train_edge_index_all, 1, batch_perm)
        train_edge_types_batch = train_edge_types_all[batch_perm]

        # add negative samples
        train_edge_index_batch_neg, train_edge_types_batch_neg, train_neg_sample_mask_graph_batch_neg = negative_sampling(
            train_edge_index_batch, train_edge_types_batch, data.num_nodes, omega=1.0)

        if batch_size:
            train_edge_index_batch = train_edge_index_batch[:, :int(batch_size / 2)]
            train_edge_types_batch = train_edge_types_batch[:int(batch_size / 2)]

        train_edge_index_batch = torch.cat((train_edge_index_batch, train_edge_index_batch[[1, 0]]), 1)
        train_edge_types_batch = torch.cat((train_edge_types_batch, train_edge_types_batch + data.num_relations), 0)

        model = model.to(device)

        z = model.encode(x=x.to(device), edge_index=train_edge_index_batch.to(device),
                         edge_type=train_edge_types_batch.to(device))

        probability = model.decode(z, train_edge_index_batch_neg.to(device), train_edge_types_batch_neg.to(device))

        # compute loss for positive as well as for negative edge examples
        loss = F.binary_cross_entropy_with_logits(probability, train_neg_sample_mask_graph_batch_neg.type(
            torch.float).to(device)) + reg_ratio * (
                           torch.mean(z.pow(2)) + torch.mean(model.decoder.relation_embedding.pow(2)))

        print(epoch, ':', loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            with torch.no_grad():
                # perform evaluation on cpu due to high memory consumption caused by large number of edges
                device = torch.device('cpu')
                model = model.to(device)
                model.eval()

                train_edge_index = torch.cat((data.train_edge_index, data.train_edge_index[[1, 0]]), 1)
                train_edge_type = torch.cat((data.train_edge_type, data.train_edge_type + data.num_relations), 0)

                print('batch size', train_edge_index.size())
                z = model.encode(x=x.to(device), edge_index=train_edge_index.to(device),
                                 edge_type=train_edge_type.to(device))

                edge_index_all = torch.cat(
                    (data.test_edge_index,
                     data.train_edge_index,
                     data.val_edge_index), dim=1)
                edge_type_all = torch.cat((data.test_edge_type,
                                           data.train_edge_type,
                                           data.val_edge_type))

                all_triplets = torch.stack((edge_index_all[0], edge_type_all, edge_index_all[1]), 1)
                test_triplets = torch.stack((data.test_edge_index[0], data.test_edge_type, data.test_edge_index[1]), 1)

                scores = evaluate_relational_link_prediction(latent_node_variables=z,
                                                             w=model.decoder.relation_embedding,
                                                             test_triplets=test_triplets, all_triplets=all_triplets)

                print(scores)



   
        
        
        
        


        
        
        
        
        
        
