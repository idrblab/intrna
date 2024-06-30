#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:16 2021

@author: wangyunxia@zju.edu.cn

"""

from sklearn import metrics
from tensorflow import keras
from tensorflow.python.client import device_lib

from cbks2 import CLA_EarlyStoppingAndPerformance
from net2 import MolMapDualPathNet

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
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
import time
from sklearn.utils import shuffle

class CategoricalAccuracy(keras.metrics.Metric):
    def __init__(self, name="Categorical_Accuracy", **kwargs):
        super(CategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        cutoff = 0.5
        # y_decide = tf.argmax(y_pred, axis=1)
        y_decide = []
        for x in y_pred[:,1]:
            if x >= cutoff:
                y_decide.append(1)
            else:
                y_decide.append(0)

        y_pred = tf.reshape(y_decide, shape=(-1, 1))
        
        con_matrix = metrics.confusion_matrix(y_true[:,1], y_decide)
        TN = con_matrix[0][0]
        FP = con_matrix[0][1]
        FN = con_matrix[1][0]
        TP = con_matrix[1][1]
        P = TP + FN
        N = TN + FP
        Acc = (TP + TN) / (P + N) if (P + N) > 0 else 0

        self.accuracy = Acc

    def result(self):
        return self.accuracy

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.accuracy.assign(0.0)

def calc_metrics(y_label, y_proba,y_predict):
    con_matrix = metrics.confusion_matrix(y_label[:, 1], y_predict[:, 1])
    TN = con_matrix[0][0]
    FP = con_matrix[0][1]
    FN = con_matrix[1][0]
    TP = con_matrix[1][1]
    P = TP + FN
    N = TN + FP
    Sn = TP / P if P > 0 else 0
    Sp = TN / N if N > 0 else 0
    Acc = (TP + TN) / (P + N) if (P + N) > 0 else 0
    Pre = (TP) / (TP + FP) if (TP+FP) > 0 else 0
    MCC = 0
    tmp = math.sqrt((TP + FP) * (TP + FN)) * math.sqrt((TN + FP) * (TN + FN))
    if tmp != 0:
        MCC = (TP * TN - FP * FN) / tmp
    fpr, tpr, thresholds = metrics.roc_curve(y_label[:, 1], y_proba[:, 1])
    f1score = f1_score(y_label[:, 1], y_predict[:, 1], average='binary')
    AUC = metrics.auc(fpr, tpr)
    return Acc, Sn, Sp, Pre, MCC, AUC, f1score

def Rdsplit(df, random_state = 888, split_size = [0.8, 0.1, 0.1]):
    base_indices = np.arange(len(df))
    base_indices = shuffle(base_indices, random_state = random_state)
    nb_test = int(len(base_indices) * split_size[2])
    nb_val = int(len(base_indices) * split_size[1])
    test_idx = base_indices[0:nb_test]
    valid_idx = base_indices[(nb_test):(nb_test+nb_val)]
    train_idx = base_indices[(nb_test+nb_val):len(base_indices)]
    print(len(train_idx), len(valid_idx), len(test_idx))
    return train_idx, valid_idx, test_idx

def train_model(X_feature_0, X_feature_1, y_train_valid_path, X_test_0_path, X_test_1_path, y_test_path, outpath, fold_select = 1):
  
    # load the transformed Y label
    train_validY = pd.read_csv(y_train_valid_path)
    train_valid_Y = pd.get_dummies(train_validY['label']).values
    print('Y.shape: {}'.format(train_valid_Y.shape))  # (1282, 2)
    
    # split the data into training and validation dataset
    train_idx, valid_idx, test_idx = Rdsplit(train_validY, random_state = 888, split_size = [0.8, 0.2, 0])

    trainX_0 = X_feature_0[train_idx]    
    validX_0 = X_feature_0[valid_idx]    

    trainX_1 = X_feature_1[train_idx]    
    validX_1 = X_feature_1[valid_idx]  

    trainY = train_valid_Y[train_idx]
    validY = train_valid_Y[valid_idx] 

    trainX = (trainX_0, trainX_1) 
    validX = (validX_0, validX_1)

    print(trainY.shape)
    print(validY.shape)
    print('trainX.shape: {}'.format(trainX.shape))
    print('validX.shape: {}'.format(validX.shape))

    rnamap1_size = trainX_0.shape[1:]
    rnamap2_size = trainX_1.shape[1:]

    # model parameters
    batch_size = 64
    epochs = 300 
    patience = 30
    lr = 0.0001
    weight_decay = 1e-4
    monitor = 'val_auc'
    metric = 'ROC'
    dense_layers = [128, 64]
    dense_avf = 'relu'
    last_avf = 'softmax'

    # define your model
    clf = MolMapDualPathNet(molmap1_size, molmap2_size, 
                                n_outputs=trainY.shape[-1], 
                                conv1_kernel_size = 13,
                                dense_layers=dense_layers, 
                                dense_avf = dense_avf, 
                                last_avf=last_avf)

    opt = tf.keras.optimizers.Adam(lr= lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay= weight_decay) #
    clf.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    performance = CLA_EarlyStoppingAndPerformance((trainX, trainY), 
                                                            (validX, validY), 
                                                            patience = patience,
                                                            metric = metric,
                                                            criteria = monitor)
    # fit your model
    clf.fit(trainX, trainY, batch_size= batch_size, 
                            epochs= epochs, verbose= 1, shuffle = True, 
                            validation_data = (validX, validY), 
                            callbacks=[performance]) 

    print('Best epochs: %.2f, Best loss: %.2f' % (performance.best_epoch, performance.best))




    res_eval = {}
    # 5 fold cross validation
    c_v = 5
    cv = StratifiedKFold(n_splits=c_v,random_state=1234,shuffle=True)
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_feature_1, train_validY['label'])):    
        print('the %s fold is starting' % fold)
        print('fold_select:{}'.format(fold_select))
        fold_select = int(fold_select)
        if fold == fold_select:
            trainX_0 = X_feature_0[train_idx]    
            validX_0 = X_feature_0[valid_idx]    

            trainX_1 = X_feature_1[train_idx]    
            validX_1 = X_feature_1[valid_idx]  

            trainY = train_valid_Y[train_idx]
            validY = train_valid_Y[valid_idx] 

            trainX = (trainX_0, trainX_1) 
            validX = (validX_0, validX_1)


            print(trainY.shape)
            print(validY.shape)
            print('trainX_1.shape: {}'.format(trainX_1.shape))

            molmap1_size = trainX_0.shape[1:]
            molmap2_size = trainX_1.shape[1:]

            # Hyperparameter Optimization
            learnrantes = [0.0001]
            batch_sizes = [64]
            parameters_all = [[y,z] for y in learnrantes for z in batch_sizes] 
            parameters = parameters_all[0:1]

            res_dict = {}

            for index, parameter in enumerate(parameters):
                print('The {} group parameters is: {}'.format(index, parameter))
                time_start = time.time()

                # model parameters
                batch_size = parameter[1]
                epochs = 300 #800
                patience = 30 #50 early stopping
                lr = parameter[0]
                weight_decay = 1e-4
                monitor = 'val_auc'
                metric = 'ROC'
                dense_layers = [128, 64]
                dense_avf = 'relu'
                last_avf = 'softmax'

                # define your model
                # encoders_pro = get_auto_encoders(trainX)
                clf = MolMapDualPathNet(molmap1_size, molmap2_size, 
                                            n_outputs=trainY.shape[-1], 
                                            conv1_kernel_size = 13,
                                            dense_layers=dense_layers, 
                                            dense_avf = dense_avf, 
                                            last_avf=last_avf)

                opt = tf.keras.optimizers.Adam(lr= lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay= weight_decay) #
                clf.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
                performance = CLA_EarlyStoppingAndPerformance((trainX, trainY), 
                                                                        (validX, validY), 
                                                                        patience = patience,
                                                                        metric = metric,
                                                                        criteria = monitor)
                # fit your model
                clf.fit(trainX, trainY, batch_size= batch_size, 
                                        epochs= epochs, verbose= 1, shuffle = True, 
                                        validation_data = (validX, validY), 
                                        callbacks=[performance]) 

                print('Best epochs: %.2f, Best loss: %.2f' % (performance.best_epoch, performance.best))
                
                
                train_process = pd.DataFrame(performance.history)
                
                y_proba_train = clf.predict(trainX)
                y_group_train = np.round(y_proba_train)
                Acc_train, Sn_train, Sp_train, Pre_train, MCC_train, AUC_train, f1score_train = calc_metrics(trainY, y_proba_train,y_group_train)

                y_proba_valid = clf.predict(validX)
                y_group_valid = np.round(y_proba_valid)
                Acc_valid, Sn_valid, Sp_valid, Pre_valid, MCC_valid, AUC_valid, f1score_valid = calc_metrics(validY, y_proba_valid,y_group_valid)

                
                y_proba = clf.predict(testX)
                y_group = np.round(y_proba)
                Acc, Sn, Sp, Pre, MCC, AUC, f1score  = calc_metrics(testY, y_proba,y_group)

                y_proba_df = pd.DataFrame(y_proba)
                y_group_df = pd.DataFrame(y_group)
                testY_df = pd.DataFrame(testY)
                print(Acc, Sn, Sp, Pre, MCC, AUC)
                save_path = os.path.join(outpath,'{}-fold'.format(fold))
                os.makedirs(save_path, exist_ok=True)
                y_proba_file = os.path.join(outpath,'{}-fold'.format(fold), 'y_proba.csv')
                y_proba_df.to_csv(y_proba_file)
                y_group_file = os.path.join(outpath,'{}-fold'.format(fold), 'y_group.csv')
                y_group_df.to_csv(y_group_file)
                testY_file = os.path.join(outpath,'{}-fold'.format(fold), 'testY.csv')
                testY_df.to_csv(testY_file)
                
                parameter_new = map(lambda x:str(x), parameter)
                parameter_new_str = '_'.join(parameter_new)
                model_path = os.path.join(outpath,'{}-fold'.format(fold), 'Para_' + parameter_new_str + 'bestmodel')
                os.makedirs(model_path, exist_ok=True)
                clf.save(model_path, save_format='tf')
                # res_dict[index] = [parameter, Acc_valid, Sn_valid, Sp_valid, Pre_valid, MCC_valid, AUC_valid,f1score_valid, Acc, Sn, Sp, Pre, MCC, AUC, f1score]
                times = (time.time() - time_start)/60
                res_dict[index] = [parameter, times,Sp_train, Sn_train, Pre_train, Acc_train, f1score_train,AUC_train, MCC_train, Sp_valid, Sn_valid, Pre_valid, Acc_valid, f1score_valid, AUC_valid, MCC_valid, Sp, Sn,  Pre, Acc, f1score, AUC, MCC]
                indexs = ['parameter','times (mins)','Sp_train','Sn_train','Pre_train','Acc_train','f1score_train','AUC_train','MCC_train','Sp_valid','Sn_valid','Pre_valid','Acc_valid','f1score_valid','AUC_valid','MCC_valid','Sp','Sn',' Pre','Acc','f1score','AUC','MCC']
                df_process = pd.DataFrame(res_dict,index = indexs)
                df_process_T = pd.DataFrame(df_process.values.T, index=df_process.columns, columns=df_process.index)
                df_process_T.to_csv(os.path.join(save_path, 'Result_parameters'  + parameter_new_str + '.csv'))
                train_process.to_csv(os.path.join(save_path, 'Epoch_Para_' + parameter_new_str + '.csv'))
            # 保存所有参数的值并根据测试集上MCC的值获得最优参数
            df = pd.DataFrame(res_dict,index = indexs)
            df_T = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
            df_T.to_csv(os.path.join(save_path, str(fold) + '-fold_parameters.csv'))
            df_list = df.iloc[14,:].tolist()
            maxindex = df_list.index(max(df_list))
            dfres = df.iloc[:,maxindex].tolist()
            print('The best parameter is learning rate, batchsize: {}'.format(dfres[0]))
            # 保存这一折的最好结果
            res_eval[fold] = dfres
            df_res = pd.DataFrame(res_eval,index = indexs)
            df_res.to_csv(os.path.join(save_path, str(fold) + '-fold.csv'))
            del trainX_1, validX_1
    

def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # print(device_lib.list_local_devices())
    # ######## Second step start: train model with transformed X and Y ----------------------------------
    datapath = '/public/home/wangyx/01_MolMap/Data/CPPred/human_training_test/transform_5species'
    X_feature_0_path = os.path.join(datapath,'Descriptors_56478_772.npy')
    X_feature_1_path = os.path.join(datapath,'gaps_56478_725.npy')
    y_train_valid_path = os.path.join(datapath,'Y_56478.csv')
    
    X_test_0_path = os.path.join(datapath,'Descriptors_16768_772.npy')
    X_test_1_path = os.path.join(datapath,'gaps_16768_725.npy')
    y_test_path = os.path.join(datapath,'Y_16768.csv')
    outpath = '/public/home/wangyx/01_MolMap/output_CNN_gap/Result_dual_human_5species_map'

    # 外部输入参数计算----------------------------
    parser = agp.ArgumentParser()
    parser.add_argument('-fold','--fold',help="the fold training")
    args = parser.parse_args()
    train_model(X_feature_0_path, X_feature_1_path, y_train_valid_path, X_test_0_path, X_test_1_path, y_test_path, outpath, fold_select = args.fold)






if __name__ == '__main__':
    main()

