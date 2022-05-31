import time
import numpy as np
import tensorflow as tf
import sys

from models import GAT
from utils import process
import scipy as sc
from scipy import stats

from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def one_pass(val_idx, inicount, dataset):
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # checkpt_file = 'pre_trained/cora/mod_cora.ckpt'

    basef = 0
    if dataset == 'citeseer':
        basef = 0.9
        NCL = 6
    elif dataset == 'cora':
        basef = 0.995
        NCL = 7
    elif dataset == 'pubmed':
        basef = 0.995
        NCL = 3
    else:
        print('Error! Have to set basef first at line 113 in train_entropy_density_graphcentral_ts.py!')
        NCL = 0
        sys.exit()

    NL = NCL * 20

    # training params
    batch_size = 1
    nb_epochs = 300
    patience = 100
    lr = 0.02  # learning rate 0.005
    l2_coef = 0  # weight decay 0.0005
    hid_units = [8] # numbers of hidden units per each attention head in each layer
    n_heads = [8, 1] # additional entry for the output layer
    residual = False
    nonlinearity = tf.nn.elu
    model = GAT

    print('Dataset: ' + dataset)
    print('----- Opt. hyperparams -----')
    print('lr: ' + str(lr))
    print('l2_coef: ' + str(l2_coef))
    print('----- Archi. hyperparams -----')
    print('nb. layers: ' + str(len(hid_units)))
    print('nb. units per layer: ' + str(hid_units))
    print('nb. attention heads: ' + str(n_heads))
    print('residual: ' + str(residual))
    print('nonlinearity: ' + str(nonlinearity))
    print('model: ' + str(model))

    # adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, labels = \
        process.load_data_AL(dataset, val_idx, inicount)

    features, spars = process.preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = y_train.shape[1]

    adj = adj.todense()

    features = features[np.newaxis]
    adj = adj[np.newaxis]
    y_train = y_train[np.newaxis]
    y_val = y_val[np.newaxis]
    y_test = y_test[np.newaxis]
    train_mask = train_mask[np.newaxis]
    val_mask = val_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]

    biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

    # calculate the percentage of elements smaller than the k-th element
    def perc(input,k): return sum([1 if i else 0 for i in input<input[k]])/float(len(input))

    # calculate the percentage of elements larger than the k-th element
    def percd(input,k): return sum([1 if i else 0 for i in input>input[k]])/float(len(input))

    normcen = np.loadtxt("res/"+ dataset +"/graphcentrality/normcen")
    cenperc = np.asarray([perc(normcen,i) for i in range(len(normcen))])

    with tf.Graph().as_default():
        with tf.name_scope('input'):
            ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size), name='feature')
            bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes), name='bias')
            lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes), name='label')
            msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes), name='mask')
            attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
            ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
            is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')

        logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                    attn_drop, ffd_drop,
                                    bias_mat=bias_in,
                                    hid_units=hid_units, n_heads=n_heads,
                                    residual=residual, activation=nonlinearity)
        log_resh = tf.reshape(logits, [-1, nb_classes])
        lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])
        loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

        pred = tf.nn.softmax(logits[0])

        train_op = model.training(loss, lr, l2_coef)

        saver = tf.train.Saver()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        vlss_mn = np.inf
        vacc_mx = 0.0
        curr_step = 0

        centroids = None

        with tf.Session() as sess:
            sess.run(init_op)

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

            for epoch in range(nb_epochs):
                gamma = np.random.beta(1, 1.005 - basef ** epoch)
                # alpha = beta = charlie = (1 - gamma) / 3
                alpha = beta = (1 - gamma) / 2

                '''
                tr_step = 0
                tr_size = features.shape[0]
                
                while tr_step * batch_size < tr_size:
                    _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                        feed_dict={
                            ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                            bias_in: biases[tr_step*batch_size:(tr_step+1)*batch_size],
                            lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                            msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                            is_train: True,
                            attn_drop: 0.6, ffd_drop: 0.6})
                    train_loss_avg += loss_value_tr
                    train_acc_avg += acc_tr
                    tr_step += 1
                '''

                _, loss_value_tr, acc_tr, embs = sess.run([train_op, loss, accuracy, pred],
                                                          feed_dict={
                                                                    ftr_in: features,
                                                                    bias_in: biases,
                                                                    lbl_in: y_train,
                                                                    msk_in: train_mask,
                                                                    is_train: True,
                                                                    attn_drop: 0.6, ffd_drop: 0.6})
                # embs = embs[0]
                if len(idx_train) < NL:
                    entropy = sc.stats.entropy(embs.T)
                    train_mask = process.sample_mask(idx_train, labels.shape[0])
                    # entropy[train_mask+val_mask+test_mask]=-100
                    entrperc = np.asarray([perc(entropy, i) for i in range(len(entropy))])
                    kmeans = KMeans(n_clusters=NCL, random_state=0).fit(embs)
                    ed = euclidean_distances(embs, kmeans.cluster_centers_)
                    # the larger ed_score is, the far that node is away from cluster centers,
                    # the less representativeness the node is
                    ed_score = np.min(ed, axis=1)
                    edprec = np.asarray([percd(ed_score, i) for i in range(len(ed_score))])

                    '''
                    pseudo_labels, centroids = process.alignment(centroids, kmeans, NCL)
                    losses = sess.run([sample_losses],
                                     feed_dict={
                                         ftr_in: features,
                                         bias_in: biases,
                                         lbl_in: pseudo_labels[np.newaxis],
                                         msk_in: np.asarray([True] * pseudo_labels.shape[0])[np.newaxis],
                                         is_train: False,
                                         attn_drop: 0.0, ffd_drop: 0.0})
                    lossperc = np.asarray([perc(losses[0], i) for i in range(len(losses[0]))])
                    '''

                    # finalweight = alpha * entrperc + beta * edprec + gamma * cenperc + charlie * lossperc
                    finalweight = alpha * entrperc + beta * edprec + gamma * cenperc

                    finalweight[train_mask + val_mask[0] + test_mask[0]] = -100
                    select = np.argmax(finalweight)
                    idx_train.append(select)
                    train_mask = process.sample_mask(idx_train, labels.shape[0])
                    y_train = np.zeros(labels.shape)
                    y_train[train_mask, :] = labels[train_mask, :]

                    train_mask = train_mask[np.newaxis]
                    y_train = y_train[np.newaxis]
                else:
                    print('finish select!')

                '''
                vl_step = 0
                vl_size = features.shape[0]
    
                while vl_step * batch_size < vl_size:
                    loss_value_vl, acc_vl = sess.run([loss, accuracy],
                        feed_dict={
                            ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                            bias_in: biases[vl_step*batch_size:(vl_step+1)*batch_size],
                            lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                            msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                            is_train: False,
                            attn_drop: 0.0, ffd_drop: 0.0})
                    val_loss_avg += loss_value_vl
                    val_acc_avg += acc_vl
                    vl_step += 1
                '''
                loss_value_vl, acc_vl = sess.run([loss, accuracy],
                                                 feed_dict={
                                                     ftr_in: features,
                                                     bias_in: biases,
                                                     lbl_in: y_val,
                                                     msk_in: val_mask,
                                                     is_train: False,
                                                     attn_drop: 0.0, ffd_drop: 0.0})

                print('Epoch ' + str(epoch) + ': Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                        (loss_value_tr, acc_tr,
                        loss_value_vl, acc_vl))

                '''
                if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                    if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                        vacc_early_model = val_acc_avg/vl_step
                        vlss_early_model = val_loss_avg/vl_step
                        saver.save(sess, checkpt_file)
                    vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                    vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                    curr_step = 0
                else:
                    curr_step += 1
                    if curr_step == patience:
                        print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                        print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                        break
    
                train_loss_avg = 0
                train_acc_avg = 0
                val_loss_avg = 0
                val_acc_avg = 0
                '''

            # saver.restore(sess, checkpt_file)

            '''
            ts_size = features.shape[0]
            ts_step = 0
            ts_loss = 0.0
            ts_acc = 0.0
    
            while ts_step * batch_size < ts_size:
                loss_value_ts, acc_ts = sess.run([loss, accuracy],
                    feed_dict={
                        ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                        bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
                        lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                        msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                ts_loss += loss_value_ts
                ts_acc += acc_ts
                ts_step += 1
    
            print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)
            '''

            lo = sess.run([pred],
                          feed_dict={
                             ftr_in: features,
                             bias_in: biases,
                             lbl_in: y_test,
                             msk_in: test_mask,
                             is_train: False,
                             attn_drop: 0.0, ffd_drop: 0.0})[0]

            y_true = np.argmax(y_test[0], axis=1)[test_mask[0]]
            y_pred = np.argmax(lo, axis=1)[test_mask[0]]
            macrof1 = f1_score(y_true, y_pred, average='macro')
            microf1 = f1_score(y_true, y_pred, average='micro')

            print("macro {}".format(macrof1))
            print("micro {}".format(microf1))
            sess.close()

            return macrof1, microf1


if __name__ == '__main__':
    macro, micro = [], []
    for val_idx in range(5):
        mac, mic = one_pass(val_idx, 4, sys.argv[1])
        macro.append(mac)
        micro.append(mic)
    print("Average of macrof1 is {}".format(sum(macro) / 5))
    print("Average of microf1 is {}".format(sum(micro) / 5))
