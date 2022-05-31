from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import scipy as sc
import sys
import os
import networkx as nx

from gcn.utils import *
from gcn.models import GCN, MLP
from utils import load_randomalpdata
from utils import sample_mask
import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score
# from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelBinarizer

import torch
from kmeans_pytorch import kmeans

if(len(sys.argv)<5):
    print('Error! Please refer README file for the argument setting!')
    sys.exit()

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

dataset_str = sys.argv[4]
#time sensitive weight
basef = 0
if dataset_str == 'citeseer':
    basef = 0.9
elif dataset_str == 'cora':
    basef = 0.995
elif dataset_str == 'pubmed':
    basef = 0.995
if(basef==0):
    print('Error! Have to set basef first at line 113 in mine.py!')
    sys.exit()

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', sys.argv[4], 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, labels, graph = load_randomalpdata(FLAGS.dataset, sys.argv[1], int(sys.argv[2]))

print(features.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)
print(labels.shape)
print(sum(test_mask))
print(idx_train)
print(labels.shape)
print(adj.shape)

# Some preprocessing
features = preprocess_features(features)
#exit(0)

if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

NCL = int(sys.argv[3])
NL = NCL*20

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Define model evaluation function, add predicted results so as to calculate macro-f1 and micro-f1
def evaluatepred(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.predict()], feed_dict=feed_dict_val)
    predlabels=np.argmax(outs_val[1],axis=1)
    return outs_val[0], predlabels, (time.time() - t_test)


# calculate the percentage of elements smaller than the k-th element
def perc(input,k): return sum([1 if i else 0 for i in input<input[k]])/float(len(input))


# calculate the percentage of elements larger than the k-th element
def percd(input,k): return sum([1 if i else 0 for i in input>input[k]])/float(len(input))


def alignment(centroids, km, NCL):
    if centroids is not None:

        old_centroids = centroids
        new_centroids = km.cluster_centers_

        DistanceMatrix = np.linalg.norm(old_centroids[:, np.newaxis, :] - new_centroids[np.newaxis, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(DistanceMatrix)

        alignment_labels = list(col_ind)
        for i in range(NCL):
            label = alignment_labels[i]
            centroids[i] = new_centroids[label]

        pseudo2label = {label: i for i, label in enumerate(alignment_labels)}
        pseudo_labels = np.array([pseudo2label[label] for label in km.labels_])

    else:
        centroids = km.cluster_centers_
        pseudo_labels = km.labels_

    lb = LabelBinarizer()
    pseudo_labels = lb.fit_transform(pseudo_labels)

    return pseudo_labels, centroids


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

normcen = np.loadtxt("res/"+FLAGS.dataset+"/graphcentrality/normcen")
cenperc = np.asarray([perc(normcen,i) for i in range(len(normcen))])


# Pretrain
for e in range(10):
    t = time.time()
    
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.predict()], feed_dict=feed_dict)

    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (e + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    
print("Pretraining Finished")

centroids = None

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()

    # time sensitive parameters
    gamma = np.random.beta(1, 1.005-basef**epoch)
    alpha = beta = (1-gamma)/2

    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.predict(), model.outputs], feed_dict=feed_dict)


    if len(idx_train)<NL:
        entropy = sc.stats.entropy(outs[3].T)
        train_mask = sample_mask(idx_train, labels.shape[0])
        #entropy[train_mask+val_mask+test_mask]=-100
        entrperc = np.asarray([perc(entropy,i) for i in range(len(entropy))])

        torch_out = torch.from_numpy(outs[3])
        # kmeans = KMeans(n_clusters=NCL, random_state=0).fit(outs[3])
        _, centroids = kmeans(
            X=torch_out, num_clusters=NCL, distance='euclidean', device=torch.device('cuda:0')
        )
        ed=euclidean_distances(torch_out.cpu(),centroids.cpu())
        ed_score = np.min(ed,axis=1)	#the larger ed_score is, the far that node is away from cluster centers, the less representativeness the node is
        edprec = np.asarray([percd(ed_score,i) for i in range(len(ed_score))])

        '''
        pseudo_labels, centroids = alignment(centroids, kmeans, NCL)
        feed_dict = construct_feed_dict(features, support, pseudo_labels, [True] * pseudo_labels.shape[0], placeholders)
        sample_loss = sess.run([model.sample_losses()], feed_dict=feed_dict)[0]
        lossperc = np.asarray([perc(sample_loss, i) for i in range(len(sample_loss))])
        '''

        finalweight = alpha*entrperc + beta*edprec + gamma*cenperc
        finalweight[train_mask+val_mask+test_mask]=-100
        select=np.argmax(finalweight)
        idx_train.append(select)
        train_mask = sample_mask(idx_train, labels.shape[0])
        y_train = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
    else:
        print ('finish select!')


    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, y_pred, test_duration = evaluatepred(features, support, y_test, test_mask, placeholders)
y_true=np.argmax(y_test,axis=1)[test_mask]
macrof1 = f1_score(y_true, y_pred[test_mask], average='macro')
microf1 = f1_score(y_true, y_pred[test_mask], average='micro')

print("macro {}".format(macrof1))
print("micro {}".format(microf1))

directory = "res/"+FLAGS.dataset+"/"
if not os.path.exists(directory):
    os.makedirs(directory)

f = open(directory+"val_"+sys.argv[1]+"_ini_"+sys.argv[2]+"_macrof1.txt","a");
f.write("{:.5f}\n".format(macrof1))
f.close()
f1 = open(directory+"/val_"+sys.argv[1]+"_ini_"+sys.argv[2]+"_microf1.txt","a");
f1.write("{:.5f}\n".format(microf1))
f1.close()