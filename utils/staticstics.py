from sklearn import metrics
from tensorflow import keras
from tensorflow.python.client import device_lib


from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
import math
import tensorflow as tf
tf.random.set_seed(888)
from functools import reduce
from tqdm import tqdm
tqdm.pandas(ascii=True)
import argparse as agp

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp

def get_Fruit_fly_label(hg38fasta):
    seq_names = []
    types = {}

    with open(hg38fasta, 'r') as f:
        for line in f:
            if line[0] == '>':
                line = line.strip()
                seq_names.append(line)
                seq_name = line.split(' ')[0]
                type = line.split(' ')[1]
                types[seq_name] = type
    return types

def to_categorical(y, num_classes=None, dtype='float64'):
    y = np.array(y, dtype='int')

    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]

    categorical = np.zeros((n, num_classes), dtype=dtype)

    categorical[np.arange(n), y] = 1

    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
# 多分类评估指标
def calculate_multilabel_auc(true_label_, pre_label, logits_, class_list, path, n_name,
                             title='ROC Curve of Deep Neural Network'):
    colors = ['#437A8B', '#C23147', '#5F86CC', '#F09150', '#AA65C7', '#E68223', '#D52685', '#EF7670', '#00A4C5',
              '#9184C1', '#FF9900', '#BEDFB8', '#60C1BD', '#00704A', '#CEFFCE', '#28FF28', '#007500', '#FFFF93',
              '#8C8C00', '#FFB5B5']
    true_label = to_categorical(true_label_)
    pre_label = np.array(pre_label)
    confusion = metrics.confusion_matrix(true_label_, logits_)

    FP = confusion.sum(axis=0) - np.diag(confusion)  
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion )
    TN = confusion.sum() - (FP + FN + TP)
    SPE = 0
    for i in range(len(class_list)):
        spe = TN[i]/(TN[i]+FP[i])
        SPE = SPE + spe
    SPE = SPE/len(class_list)

    plt.figure(figsize=(10, 10))
    plt.rcParams['savefig.dpi'] = 300  
    plt.rcParams['figure.dpi'] = 300 
    plt.rcParams.update({'font.size': 10})

    bwith = 2.0  
    TK = plt.gca()  
    TK.spines['bottom'].set_linewidth(bwith)  
    TK.spines['left'].set_linewidth(bwith) 
    TK.spines['top'].set_linewidth(bwith)  
    TK.spines['right'].set_linewidth(bwith) 
    lw = 1

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(class_list)
    for i in range(0, len(class_list)):
        fpr[i], tpr[i], _ = roc_curve(true_label[:, i], pre_label[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i], marker='.', markersize=2, lw=lw, linestyle='dashed',
                 # plt.plot(fpr[i], tpr[i], color=colors[i], marker='.', markersize=2, lw=lw,
                 label=class_list[i] + ' AUC = %0.3f' % (roc_auc[i]))

    # # micro
    fpr["micro"], tpr["micro"], _ = roc_curve(true_label.ravel(), pre_label.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"], color='Red', marker='.', markersize=2, lw=lw, linestyle=':',
             label='micro AUC = %0.3f' % (roc_auc["micro"]))
    print('roc_auc["micro"]')
    print(roc_auc["micro"])
    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.plot(fpr["macro"], tpr["macro"], color='green', marker='.', markersize=2, lw=lw, linestyle=':',
             label='macro AUC = %0.3f' % (roc_auc["macro"]))

    print('roc_auc["macro"]')
    print(roc_auc["macro"])

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    # plt.title('Receiver Operating Characteristic', fontsize=25, pad=10)
    plt.title(title, fontsize=25, pad=10)
    # plt.legend(loc="lower right")
    plt.legend(frameon=False, loc="lower right", fontsize='large')

    plt.savefig(path + '/' + str(n_name) + '.png')
    plt.close()

    # Modeal Evaluation
    y_test = list(true_label_)
    predict_ = list(logits_)
    ACC = metrics.accuracy_score(y_test, predict_)
    MCC = metrics.matthews_corrcoef(y_test, predict_)
    precision = metrics.precision_score(y_test, predict_, average='macro')
    f1 = metrics.f1_score(y_test, predict_, average='macro')
    recall = metrics.recall_score(y_test, predict_, average='macro')
    fbeta = metrics.fbeta_score(y_test, predict_, average='macro', beta=0.5)

    return roc_auc["micro"], ACC, MCC, precision, f1, recall, fbeta, SPE


# combine the importance score file
def make_importance(data_path, num):
    na_lists = list(range(0,800,100))
    before = -100
    for index, na_list in enumerate(na_lists):
        gap_num = na_list-before
        filepath = data_path + str(na_list) +'.csv'
        feature_data = pd.read_csv(filepath)
        feature_data01 = feature_data['importance'][0:gap_num].to_frame()
        if index == 0:
            feature_com = feature_data01
        else:
            feature_com = pd.concat([feature_com,feature_data01],axis = 0)
        # print(feature_data01)
        # print(gap_num)
        before = na_list
    feature_data_01 = feature_data.drop(['importance','Unnamed: 0'],axis = 1)
    print('feature_data_01.shape')
    print(feature_data_01.shape)
    # print(feature_data_01)


    feature_com_01 = feature_com.iloc[0:num,:]
    print(feature_com_01.shape)
    # print(feature_com_01)
    feature_com_02 = feature_com_01.reset_index()
    feature_com_03 = feature_com_02.drop(['index'],axis = 1)
    print(feature_com_03.shape)
    feature_data_02 = pd.concat([feature_data_01,feature_com_03],axis = 1)
    return feature_data_02


