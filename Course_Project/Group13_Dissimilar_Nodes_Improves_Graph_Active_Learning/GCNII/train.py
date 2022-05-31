from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model import *
import uuid

from scipy import stats
import scipy as sc
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Load data
def one_pass(iter):
    adj, features, labels,idx_train,idx_val,idx_test = load_citation(args.data, iter=iter)
    cudaid = "cuda:"+str(args.dev)
    device = torch.device(cudaid)
    features = features.to(device)
    adj = adj.to(device)
    checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
    print(cudaid,checkpt_file)

    model = GCNII(nfeat=features.shape[1],
                    nlayers=args.layer,
                    nhidden=args.hidden,
                    nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    lamda = args.lamda,
                    alpha=args.alpha,
                    variant=args.variant).to(device)

    if args.data == 'cora':
        NL = 140
        NCL = 7
        basef = 0.995
    elif args.data == 'citeseer':
        NL = 120
        NCL = 6
        basef = 0.9
    else:
        raise NotImplementedError('Invalid dataset')

    optimizer = optim.Adam([
                            {'params':model.params1,'weight_decay':args.wd1},
                            {'params':model.params2,'weight_decay':args.wd2},
                            ],lr=args.lr)

    #calculate the percentage of elements smaller than the k-th element
    def perc(input,k): return sum([1 if i else 0 for i in input<input[k]])/float(len(input))

    #calculate the percentage of elements larger than the k-th element
    def percd(input,k): return sum([1 if i else 0 for i in input>input[k]])/float(len(input))

    normcen = np.loadtxt("res/"+args.data+"/graphcentrality/normcen")
    cenperc = np.asarray([perc(normcen,i) for i in range(len(normcen))])

    def sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=bool)

    def train():
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
        loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
        loss_train.backward()
        optimizer.step()
        return loss_train.item(), acc_train.item(), output


    def validate(output):
        model.eval()
        with torch.no_grad():
            loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
            acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
            return loss_val.item(),acc_val.item()

    def metrics():
        # model.load_state_dict(torch.load(checkpt_file))
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            y_pred = torch.argmax(output[idx_test], dim=1).cpu()
            y_true = labels[idx_test].cpu()
            macrof1 = f1_score(y_true, y_pred, average='macro')
            microf1 = f1_score(y_true, y_pred, average='micro')
        return macrof1, microf1

    t_total = time.time()
    bad_counter = 0
    best = 999999999
    best_epoch = 0
    acc = 0
    softmax = torch.nn.Softmax(dim=1)
    for epoch in range(args.epochs):
        gamma = np.random.beta(1, 1.005 - basef ** epoch)
        alpha = beta = (1 - gamma) / 2

        loss_tra, acc_tra, output = train()
        loss_val, acc_val = validate(output)

        # output = softmax(output).cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        if len(idx_train) < NL:
            entropy = sc.stats.entropy(output.T)
            train_mask = sample_mask(idx_train, labels.shape[0])
            # entropy[train_mask+val_mask+test_mask]=-100
            entrperc = np.asarray([perc(entropy, i) for i in range(len(entropy))])
            kmeans = KMeans(n_clusters=NCL, random_state=0).fit(output)
            ed = euclidean_distances(output, kmeans.cluster_centers_)
            ed_score = np.min(ed,
                              axis=1)  # the larger ed_score is, the far that node is away from cluster centers, the less representativeness the node is
            edprec = np.asarray([percd(ed_score, i) for i in range(len(ed_score))])
            finalweight = alpha * entrperc + beta * edprec + gamma * cenperc
            finalweight[idx_train] = -100
            finalweight[idx_val] = -100
            finalweight[idx_test] = -100
            select = np.argmax(finalweight)
            idx_train.append(select)

        if(epoch+1)%1 == 0:
            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                'acc:{:.2f}'.format(acc_tra*100),
                '| val',
                'loss:{:.3f}'.format(loss_val),
                'acc:{:.2f}'.format(acc_val*100))
        '''
        if loss_val < best:
            best = loss_val
            best_epoch = epoch
            acc = acc_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1
    
        if bad_counter == args.patience:
            break
        '''

    mic, mac = metrics()
    print(mic, mac)
    return mic, mac

if __name__ == "__main__":
    mic_list, mac_list = [], []
    for i in range(5):
        mic, mac = one_pass(i)
        mic_list.append(mic)
        mac_list.append(mac)
    print("Average of macrof1 is {}".format(sum(mac_list) / 5))
    print("Average of microf1 is {}".format(sum(mic_list) / 5))

    





