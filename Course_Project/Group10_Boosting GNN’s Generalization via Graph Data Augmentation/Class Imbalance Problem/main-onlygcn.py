import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import models
import utils
import data_load
import random
import ipdb
import copy

import torch
from torch.utils.tensorboard import SummaryWriter

#from torch.utils.tensorboard import SummaryWriter

# Training setting
parser = utils.get_parser()

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

writer = SummaryWriter(args.run_folder)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data
if args.dataset == 'cora':
    adj, features, labels = data_load.load_data()
    class_sample_num = 20
    im_class_num = 3
elif args.dataset == 'BlogCatalog':
    adj, features, labels = data_load.load_data_Blog()
    im_class_num = 14 #set it to be the number less than 100
    class_sample_num = 20 #not used
elif args.dataset == 'twitter':
    adj, features, labels = data_load.load_sub_data_twitter()
    im_class_num = 1
    class_sample_num = 20 #not used
else:
    print("no this dataset: {args.dataset}")


#for artificial imbalanced setting: only the last im_class_num classes are imbalanced
c_train_num = []
for i in range(labels.max().item() + 1):
    if args.imbalance and i > labels.max().item()-im_class_num: #only imbalance the last classes
        c_train_num.append(int(class_sample_num*args.im_ratio))

    else:
        c_train_num.append(class_sample_num)

#get train, validation, test data split
if args.dataset == 'BlogCatalog':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)
elif args.dataset == 'cora':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_arti(labels, c_train_num)
elif args.dataset == 'twitter':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)

#method_1: oversampling in input domain
if args.setting == 'upsampling':
    adj,features,labels,idx_train = utils.src_upsample(adj,features,labels,idx_train,portion=args.up_scale, im_class_num=im_class_num)
if args.setting == 'smote':
    adj,features,labels,idx_train = utils.src_smote(adj,features,labels,idx_train,portion=args.up_scale, im_class_num=im_class_num)



    
encoder = models.GCN_En2(nfeat=features.shape[1],
        nhid=args.nhid,
        nembed=args.nhid,
        dropout=args.dropout)
classifier = models.Classifier(nembed=args.nhid, 
        nhid=args.nhid, 
        nclass=labels.max().item() + 1, 
        dropout=args.dropout)

decoder = models.Decoder(nembed=args.nhid,
        dropout=args.dropout)


optimizer_en = optim.Adam(encoder.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
optimizer_cls = optim.Adam(classifier.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
optimizer_de = optim.Adam(decoder.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)



if args.cuda:
    encoder = encoder.cuda()
    classifier = classifier.cuda()
    decoder = decoder.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    encoder.train()
    classifier.train()
    decoder.train()
    optimizer_en.zero_grad()
    optimizer_cls.zero_grad()
    optimizer_de.zero_grad()

    embed = encoder(features, adj)

    labels_new = labels
    idx_train_new = idx_train
    adj_new = adj

    #ipdb.set_trace()
    output = classifier(embed, adj_new)


    loss_train = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new])

    acc_train = utils.accuracy(output[idx_train], labels_new[idx_train])
    loss = loss_train
    loss_rec = loss_train

    loss.backward()
    optimizer_en.step()
    optimizer_cls.step()
    
    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = utils.accuracy(output[idx_val], labels[idx_val])

    # TensorBoard
    writer.add_scalar("Train: Loss", loss_train.item(), epoch)
    writer.add_scalar("Train: Accuracy", acc_train.item(), epoch)

    writer.add_scalar("Validation: Loss", loss_val.item(), epoch)
    writer.add_scalar("Validation: Accuracy", acc_val.item(), epoch)

    #ipdb.set_trace()
    utils.print_class_acc(output[idx_val], labels[idx_val], class_num_mat[:,1])

    print('Epoch: {:05d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_rec: {:.4f}'.format(loss_rec.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(epoch = 0):
    encoder.eval()
    classifier.eval()
    decoder.eval()
    embed = encoder(features, adj)
    output = classifier(embed, adj)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = utils.accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    # TensorBoard
    writer.add_scalar("Test: Loss", loss_test, epoch)
    writer.add_scalar("Test: Accuracy", acc_test, epoch)

    utils.print_class_acc(output[idx_test], labels[idx_test], class_num_mat[:,2], pre='test')

    '''
    if epoch==40:
        torch
    '''


def save_model(epoch):
    saved_content = {}

    saved_content['encoder'] = encoder.state_dict()
    saved_content['decoder'] = decoder.state_dict()
    saved_content['classifier'] = classifier.state_dict()

    torch.save(saved_content, 'checkpoint/{}/{}_{}_{}_{}.pth'.format(args.dataset,args.setting,epoch, args.opt_new_G, args.im_ratio))

    return

def load_model(filename):
    loaded_content = torch.load('checkpoint/{}/{}.pth'.format(args.dataset,filename), map_location=lambda storage, loc: storage)

    encoder.load_state_dict(loaded_content['encoder'])
    decoder.load_state_dict(loaded_content['decoder'])
    classifier.load_state_dict(loaded_content['classifier'])

    print("successfully loaded: "+ filename)

    return

# Train model
if args.load is not None:
    load_model(args.load)

t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)

    if epoch % 10 == 0:
        test(epoch)

    if epoch % 100 == 0:
        save_model(epoch)


print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

writer.flush()
writer.close()
