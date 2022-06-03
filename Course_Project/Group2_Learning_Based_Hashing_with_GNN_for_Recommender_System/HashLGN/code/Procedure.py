'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import pickle
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
import faiss
import time
from sklearn.metrics import roc_auc_score


CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg, f1 = [], [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
        f1.append(ret['f1'])
    #ndcg.append(0)
    return {'recall':np.array(recall),
            'precision':np.array(pre),
            'ndcg':np.array(ndcg),
            'f1':np.array(f1)}

def test_one_batch_threhold(X):
    sorted_items = X[0]
    groundTrue = X[1]
    r = utils.getLabel_threhold(groundTrue, sorted_items)
    pre, recall, ndcg, f1 = [], [], [], []
    ret = utils.RecallPrecisionF1_ATk_threhold(groundTrue, r)
    pre.append(ret['precision'])
    recall.append(ret['recall'])
    ndcg.append(utils.NDCGatK_r_threhold(groundTrue, r))
    f1.append(ret['f1'])
    #ndcg.append(0)
    return {'recall':np.array(recall),
            'precision':np.array(pre),
            'ndcg':np.array(ndcg),
            'f1':np.array(f1)}
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict # dict: {user: [items]}
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks)),
               'f1': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        start_time = time.time()
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            users_emb, items_emb = Recmodel.getUsersItemsEmbedding(batch_users_gpu)

            index = faiss.IndexFlatIP(items_emb.shape[1])
            index.add(items_emb.cpu().detach().numpy())
            faiss.normalize_L2(items_emb.cpu().detach().numpy())

            # exclude
            k = 500
            _, I_top_500 = index.search(users_emb.cpu().detach().numpy(), k)
            rating_K = np.zeros((len(users_emb), max_K))
            for i in range(len(users_emb)):
                rating_K[i] = [value for value in I_top_500[i] if value not in allPos[i]][:max_K]

            users_list.append(batch_users)
            rating_list.append(torch.from_numpy(rating_K).cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            results['f1'] += result['f1']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['f1'] /= float(len(users))

        end_time = time.time()
        print(end_time - start_time)

        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/F1@{world.topks}',
                          {str(world.topks[i]): results['f1'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results


def Test_threhold(dataset, Recmodel, epoch, w=None, multicore=0, threhold=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict  # dict: {user: [items]}
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks)),
               'f1': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())

        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        start_time = time.time()
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            # Get epoch 100's users_emb & items_emb
            # users_emb, items_emb = Recmodel.getPretrainUsersItemsEmbedding(batch_users_gpu, 100)
            users_emb, items_emb = Recmodel.getUsersItemsEmbedding(batch_users_gpu)

            index = faiss.IndexFlatIP(items_emb.shape[1])
            index.add(items_emb.cpu().detach().numpy())
            faiss.normalize_L2(items_emb.cpu().detach().numpy())

            # threhold
            lims, D, I = index.range_search(users_emb.cpu().detach().numpy(), threhold)
            rating_K = []
            for i in range(len(users_emb)):
                rating_K.append(I[lims[i]:lims[i+1]])

            # exclude
            rating_K = []
            for i in range(len(users_emb)):
                relevant_items = I[lims[i]:lims[i+1]]
                rating_K.append([value for value in relevant_items if value not in allPos[i]])

            users_list.append(batch_users)
            rating_list.append(rating_K)
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
              # threshold
              pre_results.append(test_one_batch_threhold(x))

        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            results['f1'] += result['f1']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['f1'] /= float(len(users))

        end_time = time.time()
        print(end_time - start_time)

        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/F1@{world.topks}',
                          {str(world.topks[i]): results['f1'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)

        return results, end_time - start_time


# Helper function for pretrain model
def Test_Pretrain(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict  # dict: {user: [items]}
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks)),
               'f1': np.zeros(len(world.topks))}

    with torch.no_grad():
        users = list(testDict.keys())

        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1

        start_time = time.time()
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            users_emb, items_emb = Recmodel.getPretrainUsersItemsEmbedding(batch_users_gpu, epoch)

            index = faiss.IndexFlatIP(items_emb.shape[1])
            index.add(items_emb.cpu().detach().numpy())
            faiss.normalize_L2(items_emb.cpu().detach().numpy())

            # exclude
            k = 500
            _, I_top_500 = index.search(users_emb.cpu().detach().numpy(), k)
            rating_K = np.zeros((len(users_emb), max_K))
            for i in range(len(users_emb)):
                temp = [value for value in I_top_500[i] if value not in allPos[i]]
                rating_K[i][:min(max_K, len(temp))] = temp[:min(max_K, len(temp))]

            users_list.append(batch_users)
            rating_list.append(torch.from_numpy(rating_K).cpu())
            groundTrue_list.append(groundTrue)
        end_time = time.time()
        print(end_time - start_time)

        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            results['f1'] += result['f1']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['f1'] /= float(len(users))

        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/F1@{world.topks}',
                          {str(world.topks[i]): results['f1'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)

        return results, end_time - start_time
