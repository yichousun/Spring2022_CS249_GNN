"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from config import CFG
from tqdm import tqdm
import os, shutil
from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler

from utils_kdd import init_logger, VectorDataset, dot, write_down
from utils_kdd import BinaryRegularization
from loss import MSELoss
from hash import MLPHash, MLPFunc
from decoders import WeightedInnerProductDecoder, LHTIPSDecoder


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()
        self.__init_kdd()

    def __init_kdd(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"

        if not os.path.exists(CFG.save_dir):
            os.makedirs(CFG.save_dir)
        shutil.copy('config.py', CFG.save_dir)
        shutil.copy('pre_train.py', CFG.save_dir)
        LOGGER = init_logger(CFG.save_dir + 'pre_train.log')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if CFG.loader == 'torch':
            vectors = torch.load(CFG.data_file)
        elif CFG.loader == 'pickle':
            vectors = pickle.load(open(CFG.data_file, 'rb'))
        else:
            raise RuntimeError('Unrecognized Loader: ' + CFG.loader)

        if CFG.use_item == False:
            D = vectors['W']
            Q = vectors['H']
        else:
            D = vectors['H']
            Q = vectors['W']
        print('D shape')
        print(D.shape)
        print('Q shape')
        print(Q.shape)

        D_test = D[:CFG.test_num]
        Q_test = Q[:CFG.test_num]

        train_num_D = D.shape[0] - CFG.test_num - CFG.valid_num  # 2
        train_num_Q = Q.shape[0] - CFG.test_num - CFG.valid_num  # 2

        D_train = D[CFG.test_num:CFG.test_num + train_num_D]
        Q_train = Q[CFG.test_num:CFG.test_num + train_num_Q]
        if D_train.shape[0] > 1000000:
            D_train = D_train[:1000000]
        if Q_train.shape[0] > 1000000:
            Q_train = Q_train[:1000000]

        D_dataset_train = VectorDataset(D_train, lambda x: torch.FloatTensor(x))
        Q_dataset_train = VectorDataset(Q_train, lambda x: torch.FloatTensor(x))

        print(len(D_dataset_train))
        print(len(Q_dataset_train))

        D_valid = D[CFG.test_num + train_num_D:]
        Q_valid = Q[CFG.test_num + train_num_Q:]
        D_dataset_valid = VectorDataset(D_valid, lambda x: torch.FloatTensor(x))
        Q_dataset_valid = VectorDataset(Q_valid, lambda x: torch.FloatTensor(x))

        data_hash = MLPHash(CFG.emb_len, CFG.hidden_dims, CFG.code_len, use_bn=CFG.use_bn)
        if CFG.binarize_query:
            query_func = MLPHash(CFG.emb_len, CFG.hidden_dims, CFG.code_len, use_bn=CFG.use_bn)
        else:
            query_func = MLPFunc(CFG.emb_len, CFG.hidden_dims, CFG.code_len, use_bn=CFG.use_bn)
        if CFG.decoder == 'WeightedIP':
            decoder = WeightedInnerProductDecoder(CFG.code_len)
        elif CFG.decoder == 'LH-TIPS':
            decoder = LHTIPSDecoder(CFG.code_len)
        else:
            raise NotImplementedError

        #Initialization in train
        self.data_hash = data_hash.to(self.device)
        self.query_func = query_func.to(self.device)
        self.decoder = decoder.to(self.device)

        dataloader = DataLoader(D_dataset_train, batch_size=CFG.batch_size, shuffle=True)
        queryloader = DataLoader(Q_dataset_train, batch_size=CFG.batch_size, shuffle=True)

        data_val = DataLoader(D_dataset_valid, batch_size=CFG.batch_size, shuffle=False)
        query_val = DataLoader(Q_dataset_valid, batch_size=CFG.batch_size, shuffle=False)

        criterion = nn.MSELoss()
        self.optimizer1 = torch.optim.Adam(data_hash.parameters(), lr=CFG.lr)
        self.optimizer2 = torch.optim.Adam(query_func.parameters(), lr=CFG.lr)
        self.optimizer3 = torch.optim.Adam(decoder.parameters(), lr=CFG.lr)
        self.best_valid_loss = 1000
        self.stagnant_epoch = 0

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    # Add for faiss
    def getUsersItemsEmbedding(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        return users_emb, items_emb

    # Add for faiss
    def getUsersItemsEmbeddingKDD(self, users):
        all_users, all_items = self.computer()

        users_emb = all_users[users.long()]
        items_emb = all_items

        items_emb_O = items_emb.to(self.device)
        users_emb_Q = users_emb.to(self.device)

        o_code = self.data_hash(items_emb_O)
        q_code = self.query_func(users_emb_Q)
        o_bcode = self.data_hash.binarize(o_code)
        if CFG.binarize_query == True:
            q_bcode = self.query_func.binarize(q_code)
        else:
            q_bcode = q_code

        true_pred = self.decoder(o_bcode, q_bcode) #g(c)

        return q_bcode, true_pred

    def getPretrainUsersItemsEmbedding(self, users, epoch):
        # Load KDD embeddings
        # with open("checkpointKDD/epoch"+str(epoch)+"+users_emb.pickle", "rb") as f:
        #     all_users = pickle.load(f)
        # with open("checkpointKDD/epoch"+str(epoch)+"+items_emb.pickle", "rb") as f:
        #     all_items = pickle.load(f)

        # Load Baseline embeddings
        # with open("checkpointFAISS/epoch"+str(epoch)+"+users_emb.pickle", "rb") as f:
        #     all_users = pickle.load(f)
        # with open("checkpointFAISS/epoch"+str(epoch)+"+items_emb.pickle", "rb") as f:
        #     all_items = pickle.load(f)

        with open("checkpoint/epoch"+str(epoch)+"+users_emb.pickle", "rb") as f:
            all_users = pickle.load(f)
        with open("checkpoint/epoch"+str(epoch)+"+items_emb.pickle", "rb") as f:
            all_items = pickle.load(f)

        users_emb = all_users[users.long()]
        items_emb = all_items
        return users_emb, items_emb

    def getEmbeddingKDD(self, users, pos_items, neg_items):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos_items.long(), neg_items.long())

        pos_emb_O = pos_emb.to(self.device)
        neg_emb_O = neg_emb.to(self.device)
        users_emb_Q = users_emb.to(self.device)

        pos_o_code = self.data_hash(pos_emb_O)
        neg_o_code = self.data_hash(neg_emb_O)
        q_code = self.query_func(users_emb_Q)
        pos_o_bcode = self.data_hash.binarize(pos_o_code)
        neg_o_bcode = self.data_hash.binarize(neg_o_code)
        if CFG.binarize_query == True:
            q_bcode = self.query_func.binarize(q_code)
        else:
            q_bcode = q_code

        pos_true_pred = self.decoder(pos_o_bcode, q_bcode)
        neg_true_pred = self.decoder(neg_o_bcode, q_bcode)
        return q_bcode, pos_true_pred, neg_true_pred, userEmb0, posEmb0, negEmb0
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbeddingKDD(users.long(), pos.long(), neg.long())
        # add hash function here
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
