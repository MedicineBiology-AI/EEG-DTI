from __future__ import division
from __future__ import print_function
from operator import itemgetter

import time
import os
import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn import metrics
from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing
from decagon.utility import loadData

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
import datetime
nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):

        pass

data_set = 'luo'
print('data_set',data_set)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1',64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
flags.DEFINE_integer('batch_size', 128, 'minibatch size.')
flags.DEFINE_boolean('bias', True, 'Bias term.')

# you can make a log dictionary in the ./EEG-DTI.
# then you can store the log file of every runs.
instruction = 'xxxxx'
os.makedirs('./log/'+data_set+str(nowTime)+instruction)
sys.stdout = Logger('./log/'+data_set+str(nowTime)+instruction+'/terminal.txt')


#01
AUROC_01_list = []
AUPR_01_list  = []
APatK_01_list = []
ACC_01_list   = []
F1_01_list    = []
MSE_01_list   = []
MAE_01_list   = []

#10

AUROC_10_list = []
AUPR_10_list  = []
APatK_10_list = []
ACC_10_list   = []
F1_10_list    = []
MSE_10_list   = []
MAE_10_list   = []

# RealNet
# About Drug-Drug
# 1 interaction+4 sim+ 1 for protein   6networks
drug_drug_path = './DTI_data/luo/sevenNets/mat_drug_drug.txt'
drug_drug_sim_chemical_path = './DTI_data/luo/sim_network/Sim_mat_Drugs.txt'
drug_drug_sim_interaction_path = './DTI_data/luo/sim_network/Sim_mat_drug_drug.txt'
drug_drug_sim_se_path = './DTI_data/luo/sim_network/Sim_mat_drug_se.txt'
drug_drug_sim_disease_path = './DTI_data/luo/sim_network/Sim_mat_drug_disease.txt'
drug_protein_path = './DTI_data/luo/sevenNets/mat_drug_protein.txt'

# 1interaction + 3sim +1 for drug 5 networks
# About Protein
protein_drug_path = './DTI_data/luo/sevenNets/mat_protein_drug.txt'
protein_protein_path = './DTI_data/luo/sevenNets/mat_protein_protein.txt'
protein_protein_sim_sequence_path = './DTI_data/luo/sim_network/Sim_mat_Proteins.txt'
protein_protein_sim_disease_path = './DTI_data/luo/sim_network/Sim_mat_protein_disease.txt'
protein_protein_sim_interaction_path = './DTI_data/luo/sim_network/Sim_mat_protein_protein.txt'

# About drug and protein (others)...
protein_disease_path = './DTI_data/luo/sevenNets/mat_protein_disease.txt'
drug_disease_path = './DTI_data/luo/sevenNets/mat_drug_disease.txt'
drug_sideEffect_path = './DTI_data/luo/sevenNets/mat_drug_se.txt'

# Step1:Construct the graph(read the data...)

# drug_drug_adj and protein_protein_adj combine the simNets and interactions
drug_drug_adj = loadData.Load_Drug_Adj_Togerther(drug_drug_path=drug_drug_path,
                                                 drug_drug_sim_chemical_path=drug_drug_sim_chemical_path,
                                                 drug_drug_sim_interaction_path=drug_drug_sim_interaction_path,
                                                 drug_drug_sim_se_path=drug_drug_sim_se_path,
                                                 drug_drug_sim_disease_path=drug_drug_sim_disease_path)

protein_protein_adj = loadData.Load_Protein_Adj_Togerther(protein_protein_path=protein_protein_path,
                                                          protein_protein_sim_sequence_path=protein_protein_sim_sequence_path,
                                                          protein_protein_sim_disease_path=protein_protein_sim_disease_path,
                                                          protein_protein_sim_interaction_path=protein_protein_sim_interaction_path)

drug_proten_interactions, protein_drug_interactions = loadData.load_protein_drug_interactions(path=protein_drug_path)

protein_disease_adj = loadData.load_Adj_adj(threshold=0, toone=0, draw=0, sim_path=protein_disease_path)
disease_protein_adj = loadData.load_Adj_adj_transpose(threshold=0, toone=0, draw=0, sim_path=protein_disease_path)
drug_disease_adj = loadData.load_Adj_adj(threshold=0, toone=0, draw=0, sim_path=drug_disease_path)
disease_drug_adj = loadData.load_Adj_adj_transpose(threshold=0, toone=0, draw=0, sim_path=drug_disease_path)

drug_side_effect_adj = loadData.load_Adj_adj(threshold=0, toone=0, draw=0, sim_path=drug_sideEffect_path)
side_effect_drug_adj = loadData.load_Adj_adj_transpose(threshold=0, toone=0, draw=0, sim_path=drug_sideEffect_path)

# 10 fold cross-validation
for seed in range(0,10):
    val_test_size = 0.1
    print('Current seed is :',seed)
    def get_accuracy_scores(edges_pos, edges_neg, edge_type):
        feed_dict.update({placeholders['dropout']: 0})
        feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
        feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
        feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
        rec = sess.run(opt.predictions, feed_dict=feed_dict)

        def sigmoid(x):
            return 1. / (1 + np.exp(-x))

        # Predict on test set of edges
        preds = []
        actual = []
        predicted = []
        edge_ind = 0

        # pos
        for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
            score = sigmoid(rec[u, v])
            preds.append(score)
            assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 1, 'Problem 1'

            actual.append(edge_ind)
            predicted.append((score, edge_ind))
            edge_ind += 1

        preds_neg = []

        # neg
        for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
            score = sigmoid(rec[u, v])
            preds_neg.append(score)
            assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 0, 'Problem 0'

            predicted.append((score, edge_ind))
            edge_ind += 1

        preds_all = np.hstack([preds, preds_neg])
        preds_all = np.nan_to_num(preds_all)
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

        # evatalution.....
        roc_sc = metrics.roc_auc_score(labels_all, preds_all)
        aupr_sc = metrics.average_precision_score(labels_all, preds_all)
        apk_sc = rank_metrics.apk(actual, predicted, k=10)

        return roc_sc, aupr_sc, apk_sc

    def get_final_accuracy_scores(edges_pos, edges_neg, edge_type):
        feed_dict.update({placeholders['dropout']: 0})
        feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
        feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
        feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
        rec = sess.run(opt.predictions, feed_dict=feed_dict)

        def sigmoid(x):
            return 1. / (1 + np.exp(-x))

        # Predict on test set of edges
        preds = []
        actual = []
        predicted = []
        edge_ind = 0

        # pos
        for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
            score = sigmoid(rec[u, v])
            preds.append(score)
            assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 1, 'Problem 1'

            actual.append(edge_ind)
            predicted.append((score, edge_ind))
            edge_ind += 1

        preds_neg = []
        # neg
        for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
            score = sigmoid(rec[u, v])
            preds_neg.append(score)
            assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 0, 'Problem 0'

            predicted.append((score, edge_ind))
            edge_ind += 1

        preds_all = np.hstack([preds, preds_neg])
        preds_all = np.nan_to_num(preds_all)
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

        # evatalution.....
        roc_sc = metrics.roc_auc_score(labels_all, preds_all)
        aupr_sc = metrics.average_precision_score(labels_all, preds_all)
        apk_sc = rank_metrics.apk(actual, predicted, k=50)
        FPR, TPR, thresholds = metrics.roc_curve(labels_all, preds_all)

        precision,recall ,_= metrics.precision_recall_curve(labels_all, preds_all)

        mse = metrics.mean_squared_error(labels_all, preds_all)
        mae = metrics.median_absolute_error(labels_all, preds_all)
        r2 = metrics.r2_score(labels_all, preds_all)
        np.savetxt('./log/'+data_set+str(nowTime)+instruction+'/'+str(seed)+''+str(edge_type)+'_'+'_true.txt',labels_all,fmt='%d')
        np.savetxt('./log/'+data_set+str(nowTime)+instruction+'/'+str(seed)+''+str(edge_type)+'_'+'_pred.txt', preds_all,fmt='%.3f')
        preds_all[preds_all>=0.5] = 1
        preds_all[preds_all< 0.5] = 0
        acc = metrics.accuracy_score(labels_all, preds_all)
        f1 = metrics.f1_score(labels_all, preds_all, average='macro')
        return FPR, TPR, roc_sc, \
               precision,recall,aupr_sc, \
               apk_sc , thresholds ,mse, mae,r2,acc,f1

    def construct_placeholders(edge_types):
        placeholders = {
            'batch': tf.placeholder(tf.int64, name='batch'),
            'batch_edge_type_idx': tf.placeholder(tf.int64, shape=(), name='batch_edge_type_idx'),
            'batch_row_edge_type': tf.placeholder(tf.int64, shape=(), name='batch_row_edge_type'),
            'batch_col_edge_type': tf.placeholder(tf.int64, shape=(), name='batch_col_edge_type'),
            'degrees': tf.placeholder(tf.int64),
            'dropout': tf.placeholder_with_default(0., shape=()),
        }
        placeholders.update({
            'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
            for i, j in edge_types for k in range(edge_types[i,j])})
        placeholders.update({
            'feat_%d' % i: tf.sparse_placeholder(tf.float32) for i, _ in edge_types})

        return placeholders


    # data representation
    # 0 for protein / 1 for drug / 2 for disease / 3 for side-effect
    adj_mats_orig = {
        (0, 0): [protein_protein_adj,protein_protein_adj],#type1
        (0, 1): [protein_drug_interactions],#type2
        (0, 2): [protein_disease_adj],
        
        (1, 0): [drug_proten_interactions],
        (1, 1): [drug_drug_adj, drug_drug_adj],  # type3
        (1, 2): [drug_disease_adj],
        (1, 3): [drug_side_effect_adj],
        
        (2, 0): [disease_protein_adj],
        (2, 1): [disease_drug_adj],

        (3, 1): [side_effect_drug_adj],
    }

    protein_degrees = np.array(protein_protein_adj.sum(axis=0)).squeeze()
    drug_degrees    = np.array(drug_drug_adj.sum(axis=0)).squeeze()
    disease_degrees = np.array(disease_drug_adj.sum(axis=0)).squeeze()
    side_effect_degrees = np.array(side_effect_drug_adj.sum(axis=0)).squeeze()

    degrees = {
        0: [protein_degrees,protein_degrees],
        1: [drug_degrees,drug_degrees],
        2: [disease_degrees],
        3: [side_effect_degrees]
    }

    # # featureless (genes)
    gene_feat = sp.identity(1512)
    protein_nonzero_feat, protein_num_feat = gene_feat.shape
    gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

    #
    # # features (drugs)
    drug_feat = sp.identity(708)
    # drug_feat = Drug_Drug_adj
    drug_nonzero_feat, drug_num_feat = drug_feat.shape
    drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

    # data representation

    diease_feat = sp.identity(5603)
    diease_nonzero_feat, diease_num_feat = diease_feat.shape
    diease_feat = preprocessing.sparse_to_tuple(diease_feat.tocoo())
    # NOTICE

    side_effect_feat = sp.identity(4192)
    side_effect_nonzero_feat, side_effect_num_feat = side_effect_feat.shape
    side_effect_feat = preprocessing.sparse_to_tuple(side_effect_feat.tocoo())
    extra_side_effect_feat = side_effect_feat
    # NOTICE
    
    num_feat = {
        0: protein_num_feat,
        1: drug_num_feat,
        2: diease_num_feat,
        3: side_effect_num_feat,
    }
    nonzero_feat = {
        0: protein_nonzero_feat,
        1: drug_nonzero_feat,
        2: diease_nonzero_feat,
        3: side_effect_nonzero_feat
    }
    feat = {
        0: gene_feat,
        1: drug_feat,
        2: diease_feat,
        3: side_effect_feat

    }

    edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}

    # edge_types
    # {(0, 0): 2, (0, 1): 1, (0, 2): 1, (1, 0): 1, (1, 1): 2, (1, 2): 1, (2, 0): 1, (2, 1): 1, (2, 2): 1}
    edge_type2decoder = {
        (0, 0): 'innerproduct',
        (0, 1): 'innerproduct',
        (0, 2): 'innerproduct',

        (1, 0): 'innerproduct',
        (1, 1): 'innerproduct',
        (1, 2): 'innerproduct',
        (1, 3): 'innerproduct',

        (2, 0): 'innerproduct',
        (2, 1): 'innerproduct',

        (3, 1): 'innerproduct',
    }

    edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
    print('edge_types',edge_types)
    num_edge_types = sum(edge_types.values())
    print("Edge types:", "%d" % num_edge_types)


    # Important -- Do not evaluate/print validation performance every iteration as it can take
    # substantial amount of time
    PRINT_PROGRESS_EVERY = 20

    print("Defining placeholders")
    placeholders = construct_placeholders(edge_types)

    print("Create minibatch iterator")
    minibatch = EdgeMinibatchIterator(
        adj_mats=adj_mats_orig,
        feat=feat,
        seed=seed,
        data_set = data_set,
        edge_types=edge_types,
        batch_size=FLAGS.batch_size,
        val_test_size=val_test_size
    )

    print("Create model")
    model = DecagonModel(
        placeholders=placeholders,
        num_feat=num_feat,
        nonzero_feat=nonzero_feat,
        data_set = data_set,
        edge_types=edge_types,
        decoders=edge_type2decoder,
    )

    print("Create optimizer")
    with tf.name_scope('optimizer'):
        opt = DecagonOptimizer(
            embeddings=model.embeddings,
            latent_inters=model.latent_inters,
            latent_varies=model.latent_varies,
            degrees=degrees,
            edge_types=edge_types,
            edge_type2dim=edge_type2dim,
            placeholders=placeholders,
            batch_size=FLAGS.batch_size,
            margin=FLAGS.max_margin
        )

    print("Initialize session")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    feed_dict = {}

    # Train model


    print("Train model")
    for epoch in range(FLAGS.epochs):

        minibatch.shuffle()
        itr = 0
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
            feed_dict = minibatch.update_feed_dict(feed_dict=feed_dict,dropout=FLAGS.dropout,placeholders=placeholders)

            t = time.time()

            # Training step: run single weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.batch_edge_type_idx], feed_dict=feed_dict)
            train_cost = outs[1]
            batch_edge_type = outs[2]

            if itr % PRINT_PROGRESS_EVERY == 0:
                val_auc, val_auprc, val_apk = get_accuracy_scores(
                    minibatch.val_edges, minibatch.val_edges_false,
                    minibatch.idx2edge_type[minibatch.current_edge_type_idx])

                print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), "Edge:", "%04d" % batch_edge_type,
                      "train_loss=", "{:.5f}".format(train_cost),
                      "val_roc=", "{:.5f}".format(val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
                      "val_apk=", "{:.5f}".format(val_apk), "time=", "{:.5f}".format(time.time() - t))

            itr += 1

    print("Optimization finished!")

    for et in range(num_edge_types):
        print('et=', et)
        PRINT = 1
        if PRINT==1:
            FPR, TPR, roc_score, \
            precision, recall, auprc_score, \
            apk_score, \
            thresholds, mse, mae, r2, acc, f1 = get_final_accuracy_scores(
                minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
            # if et==1 or et==2:
            # edge_types
            # {(0, 0): 2, (0, 1): 1, (0, 2): 1, (1, 0): 1, (1, 1): 2, (1, 2): 1, (2, 0): 1, (2, 1): 1, (2, 2): 1}
            if et == 2:
                AUROC_01_list.append(roc_score)
                AUPR_01_list.append(auprc_score)
                APatK_01_list.append(apk_score)
                ACC_01_list.append(acc)
                F1_01_list.append(f1)
                MSE_01_list.append(mse)
                MAE_01_list.append(mae)
            if et == 4:
                AUROC_10_list.append(roc_score)
                AUPR_10_list.append(auprc_score)
                APatK_10_list.append(apk_score)
                ACC_10_list.append(acc)
                F1_10_list.append(f1)
                MSE_10_list.append(mse)
                MAE_10_list.append(mae)

            print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
            print("Edge type:", "%04d" % et, "Test AUROC score", "{:.5f}".format(roc_score))
            print("Edge type:", "%04d" % et, "Test AUPRC score", "{:.5f}".format(auprc_score))
            print("Edge type:", "%04d" % et, "Test AP@k score", "{:.5f}".format(apk_score))
            print("Edge type:", "%04d" % et, "Test acc score", "{:.5f}".format(acc))
            print("Edge type:", "%04d" % et, "Test f1 score", "{:.5f}".format(f1))
            print("Edge type:", "%04d" % et, "Test mse score", "{:.5f}".format(mse))
            print("Edge type:", "%04d" % et, "Test mae score", "{:.5f}".format(mae))
            print("Edge type:", "%04d" % et, "Test r2 score", "{:.5f}".format(r2))
            print()

print('10-Flod-cross-val-result')


print('-----01------')
print('AUROC_01_list', AUROC_01_list)
print('AUPR_01_list', AUPR_01_list)
print('APatK_01_list', APatK_01_list)
print('ACC_01_list', ACC_01_list)
print('F1_01_list', F1_01_list)
print('MSE_01_list', MSE_01_list)
print('MAE_01_list', MAE_01_list)
print('AVG_AUROC_01_list', np.mean(AUROC_01_list).round(4))
print('AVG_AUPR_01_list', np.mean(AUPR_01_list).round(4))
print('AVG_APatK_01_list', np.mean(APatK_01_list).round(4))
print('AVG_ACC_01_list', np.mean(ACC_01_list).round(4))
print('AVG_F1_01_list', np.mean(F1_01_list).round(4))
print('AVG_MSE_01_list', np.mean(MSE_01_list).round(4))
print('AVG_MAE_01_list', np.mean(MAE_01_list).round(4))

print('-----10------')
print('AUROC_10_list', AUROC_10_list)
print('AUPR_10_list', AUPR_10_list)
print('APatK_10_list', APatK_10_list)
print('ACC_10_list', ACC_10_list)
print('F1_10_list', F1_10_list)
print('MSE_10_list', MSE_10_list)
print('MAE_10_list', MAE_10_list)
print('AVG_AUROC_10_list', np.mean(AUROC_10_list).round(4))
print('AVG_AUPR_10_list', np.mean(AUPR_10_list).round(4))
print('AVG_APatK_10_list', np.mean(APatK_10_list).round(4))
print('AVG_ACC_10_list', np.mean(ACC_10_list).round(4))
print('AVG_F1_10_list', np.mean(F1_10_list).round(4))
print('AVG_MSE_10_list', np.mean(MSE_10_list).round(4))
print('AVG_MAE_10_list', np.mean(MAE_10_list).round(4))
