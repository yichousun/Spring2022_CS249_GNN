from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import scipy as sc
import sys
import os
import networkx as nx
import gcn
import sklearn

from gcn.utils import *
from gcn.models import GCN, MLP
from utils import load_randomalpdata
from utils import load_newdata
from utils import sample_mask
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

if(len(sys.argv)<5):
    print('Error! Please refer README file for the argument setting!')
    sys.exit()

# Set random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

dataset_str = sys.argv[4]

newData= ["squirrel", "wisconsin", "cornell", "chameleon", "texas"]
#time sensitive weight
basef = 0
if dataset_str == 'citeseer':
    if sys.argv[1] != 'combined':
      basef = 0.9
    else:
      basef = 0.85
elif dataset_str == 'cora':
    if sys.argv[1] != 'combined':
      basef = 0.99
    else:
      basef = 0.95
elif dataset_str == 'pubmed':
    basef = 0.995
elif dataset_str in newData:
    basef = 0.995
if(basef==0):
    print('Error! Have to set basef first at line 113 in train_entropy_density_graphcentral_ts.py!')
    sys.exit()

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', sys.argv[4], 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

MAC = []
MIC = []
num_runs = 30 if dataset_str in ["wisconsin", "cornell", "texas"] else 10
for run in range(num_runs):
  print(run)
  if FLAGS.dataset in newData:
      adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, labels, graph = load_newdata(FLAGS.dataset, int(sys.argv[3]), int(sys.argv[2]))
  else:
      adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, labels, graph = load_randomalpdata(FLAGS.dataset, '0', int(sys.argv[2]))
  np.set_printoptions(threshold=sys.maxsize)
  message_passing = adj * adj
    

  raw_features = features.todense()
  # Some preprocessing
  print(type(features))
  features = preprocess_features(features)
  print(type(features))
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
  if dataset_str in ["squirrel", "chameleon"]:
    NL = NCL*30
  elif dataset_str in ["wisconsin", "cornell", "texas"]:
    NL = NCL*10
    FLAGS.epochs = 100
  else:
    NL = NCL*20
  tf.compat.v1.disable_eager_execution()
  # Define placeholders
  placeholders = {
      'support': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
      'features': tf.compat.v1.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
      'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, y_train.shape[1])),
      'labels_mask': tf.compat.v1.placeholder(tf.int32),
      'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
      'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)  # helper variable for sparse dropout
  }
  # Create model
  model = model_func(placeholders, input_dim=features[2][1], logging=True)

  # Initialize session
  sess = tf.compat.v1.Session()

  # Define model evaluation function
  def evaluate(features, support, labels, mask, placeholders):
      t_test = time.time()
      feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
      outs_val = sess.run([model.loss, model.accuracy, model.predict()], feed_dict=feed_dict_val)
      return outs_val[0], outs_val[1], (time.time() - t_test)

  # Define model evaluation function, add predicted results so as to calculate macro-f1 and micro-f1
  def evaluatepred(features, support, labels, mask, placeholders):
      t_test = time.time()
      feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
      outs_val = sess.run([model.loss, model.predict()], feed_dict=feed_dict_val)
      predlabels=np.argmax(outs_val[1],axis=1)
      return outs_val[0], predlabels, (time.time() - t_test)

  #calculate the percentage of elements smaller than the k-th element
  def perc(input,k): return sum([1 if i else 0 for i in input<input[k]])/float(len(input))

  #calculate the percentage of elements larger than the k-th element
  def percd(input,k): return sum([1 if i else 0 for i in input>input[k]])/float(len(input))

  # Init variables
  sess.run(tf.compat.v1.global_variables_initializer())

  cost_val = []

  normcen = np.loadtxt("res/"+FLAGS.dataset+"/graphcentrality/normcen")
  cenperc = np.asarray([perc(normcen,i) for i in range(len(normcen))])

  # Train model
  for epoch in range(FLAGS.epochs):

      t = time.time()

      #time sensitive parameters
      gamma = np.random.beta(1, 1.005-basef**epoch)

      if sys.argv[1] == 'baseline':
        alpha = beta = delta = epsilon = (1-gamma)/2
      elif sys.argv[1] == 'combined':
        alpha = beta = delta = epsilon = (1-gamma)/4
      else:
        alpha = beta = delta = epsilon = (1-gamma)/3
      # Construct feed dictionary
      feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)


      feed_dict.update({placeholders['dropout']: FLAGS.dropout})
      # Training step
      outs = sess.run([model.opt_op, model.loss, model.accuracy, model.predict(), model.outputs], feed_dict=feed_dict)

      #choose instance to label based on entropy
      if len(idx_train)<NL:
        if sys.argv[1] == 'random':
          select = np.random.randint(low=0, high=len(raw_features) - 1)
          idx_train.append(select)
          train_mask = sample_mask(idx_train, labels.shape[0])
          y_train = np.zeros(labels.shape)
          y_train[train_mask, :] = labels[train_mask, :]
        elif sys.argv[1] == 'c_only':
          finalweight = cenperc
          finalweight[train_mask+val_mask+test_mask]=-100
          select=np.argmax(finalweight)
          idx_train.append(select)
          train_mask = sample_mask(idx_train, labels.shape[0])
          y_train = np.zeros(labels.shape)
          y_train[train_mask, :] = labels[train_mask, :]
        elif sys.argv[1] == 'e_only':
          entropy = sc.stats.entropy(outs[3].T)
          train_mask = sample_mask(idx_train, labels.shape[0])
          #entropy[train_mask+val_mask+test_mask]=-100
          entrperc = np.asarray([perc(entropy,i) for i in range(len(entropy))])
          finalweight = entrperc
          finalweight[train_mask+val_mask+test_mask]=-100
          select=np.argmax(finalweight)
          idx_train.append(select)
          train_mask = sample_mask(idx_train, labels.shape[0])
          y_train = np.zeros(labels.shape)
          y_train[train_mask, :] = labels[train_mask, :]
        elif sys.argv[1] == 'fs_only':
          curr_features = raw_features[idx_train, :]
          curr_features = sklearn.preprocessing.normalize(curr_features)
          raw_features = sklearn.preprocessing.normalize(raw_features)
          similarity = []
          for i in curr_features:
            similarity.append(np.dot(raw_features, np.squeeze([i])))
          similarity = np.squeeze(np.array(similarity))
          max_similarity = np.max(similarity, axis=0)
          simprec = np.asarray([percd(max_similarity,i) for i in range(len(max_similarity))])
          finalweight = simprec
          finalweight[train_mask+val_mask+test_mask]=-100
          select=np.argmax(finalweight)
          idx_train.append(select)
          train_mask = sample_mask(idx_train, labels.shape[0])
          y_train = np.zeros(labels.shape)
          y_train[train_mask, :] = labels[train_mask, :]
        elif sys.argv[1] == 'ss_only':
          connectivity = message_passing[idx_train, :]
          max_connectivity = np.squeeze(np.array(np.max(connectivity, axis=0).todense()))
          connprec = np.asarray([percd(max_connectivity,i) for i in range(len(max_connectivity))])
          finalweight = connprec
          finalweight[train_mask+val_mask+test_mask]=-100
          select=np.argmax(finalweight)
          idx_train.append(select)
          train_mask = sample_mask(idx_train, labels.shape[0])
          y_train = np.zeros(labels.shape)
          y_train[train_mask, :] = labels[train_mask, :]
        elif sys.argv[1] == 'es_only':
          curr_embeddings = outs[4][idx_train, :]
          curr_embeddings = sklearn.preprocessing.normalize(curr_embeddings)
          raw_embeddings = sklearn.preprocessing.normalize(outs[4])
          em_similarity = []
          for i in curr_embeddings:
            em_similarity.append(np.dot(raw_embeddings, np.squeeze([i])))
          em_similarity = np.squeeze(np.array(em_similarity))
          max_em_similarity = np.max(em_similarity, axis=0)
          em_simprec = np.asarray([percd(max_em_similarity,i) for i in range(len(max_em_similarity))])
          finalweight = em_simprec
          finalweight[train_mask+val_mask+test_mask]=-100
          select=np.argmax(finalweight)
          idx_train.append(select)
          train_mask = sample_mask(idx_train, labels.shape[0])
          y_train = np.zeros(labels.shape)
          y_train[train_mask, :] = labels[train_mask, :]
        else:
          curr_features = raw_features[idx_train, :]
          curr_features = sklearn.preprocessing.normalize(curr_features)
          raw_features = sklearn.preprocessing.normalize(raw_features)
          similarity = []
          for i in curr_features:
            similarity.append(np.dot(raw_features, np.squeeze([i])))
          similarity = np.squeeze(np.array(similarity))
          max_similarity = np.max(similarity, axis=0)
          simprec = np.asarray([percd(max_similarity,i) for i in range(len(max_similarity))])

          curr_embeddings = outs[4][idx_train, :]
          curr_embeddings = sklearn.preprocessing.normalize(curr_embeddings)
          raw_embeddings = sklearn.preprocessing.normalize(outs[4])
          em_similarity = []
          for i in curr_embeddings:
            em_similarity.append(np.dot(raw_embeddings, np.squeeze([i])))
          em_similarity = np.squeeze(np.array(em_similarity))
          max_em_similarity = np.max(em_similarity, axis=0)
          em_simprec = np.asarray([percd(max_em_similarity,i) for i in range(len(max_em_similarity))])

          connectivity = message_passing[idx_train, :]
          max_connectivity = np.squeeze(np.array(np.max(connectivity, axis=0).todense()))
          connprec = np.asarray([percd(max_connectivity,i) for i in range(len(max_connectivity))])

          entropy = sc.stats.entropy(outs[3].T)
          train_mask = sample_mask(idx_train, labels.shape[0])
          #entropy[train_mask+val_mask+test_mask]=-100
          entrperc = np.asarray([perc(entropy,i) for i in range(len(entropy))])

          kmeans = KMeans(n_clusters=NCL, random_state=0).fit(outs[3])
          ed=euclidean_distances(outs[3],kmeans.cluster_centers_)
          ed_score = np.min(ed,axis=1)	#the larger ed_score is, the far that node is away from cluster centers, the less representativeness the node is
          edprec = np.asarray([percd(ed_score,i) for i in range(len(ed_score))])

          if sys.argv[1] == 'baseline':
            finalweight = alpha*entrperc + beta*edprec + gamma*cenperc
            print("entropy weight: ", alpha, " density weight: ", beta, "centrality weight: ", gamma)
          elif sys.argv[1] == 'f_similarity':
            finalweight = alpha*entrperc + beta*edprec + gamma*cenperc + delta*simprec
            print("entropy weight: ", alpha, " density weight: ", beta, " centrality weight: ", gamma, " feature similarity weight: ", delta)
          elif sys.argv[1] == 's_similarity':
            finalweight = alpha*entrperc + beta*edprec + gamma*cenperc + delta*connprec
            print("entropy weight: ", alpha, " density weight: ", beta, " centrality weight: ", gamma, " structural similarity weight: ", delta)
          elif sys.argv[1] == 'e_similarity':
            finalweight = alpha*entrperc + beta*edprec + gamma*cenperc + delta*em_simprec
            print("entropy weight: ", alpha, " density weight: ", beta, " centrality weight: ", gamma, " embedding similarity weight: ", delta)
          else:
            finalweight = alpha*entrperc + beta*edprec + gamma*cenperc + delta*connprec + epsilon*simprec
            print("entropy weight: ", alpha, " density weight: ", beta, " centrality weight: ", gamma, " stuructural similarity weight: ", delta, " feature similarity weight: ", epsilon)
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

      # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
      #     print("Early stopping...")
      #     break

  print("Optimization Finished!")

  # Testing
  test_cost, y_pred, test_duration = evaluatepred(features, support, y_test, test_mask, placeholders)
  y_true=np.argmax(y_test,axis=1)[test_mask]
  macrof1 = f1_score(y_true, y_pred[test_mask], average='macro') 
  microf1 = f1_score(y_true, y_pred[test_mask], average='micro')
  print("-------------------------------------------------------")
  print("-------------------------------------------------------")
  print("macrof1:",macrof1,"microf1:", microf1)
  directory = "res/"+FLAGS.dataset+"/"
  if not os.path.exists(directory):
      os.makedirs(directory)

  f = open(directory+"val_"+sys.argv[1]+"_ini_"+sys.argv[2]+"_macrof1.txt","a");
  f.write("{:.5f}\n".format(macrof1))
  f.close()
  f1 = open(directory+"/val_"+sys.argv[1]+"_ini_"+sys.argv[2]+"_microf1.txt","a");
  f1.write("{:.5f}\n".format(microf1))
  f1.close()
  MAC.append(macrof1)
  MIC.append(microf1)

print(MAC)
print(MIC)
import numpy as np
print("mean of macrof1 over 10 runs: ", np.mean(MAC))
print("mean of microf1 over 10 runs: ", np.mean(MIC))
print("variance of macrof1 over 10 runs: ", np.var(MAC))
print("variance of microf1 over 10 runs: ", np.var(MIC))

