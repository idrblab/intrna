import numpy as np
import pandas as pd
import random
from tqdm import tqdm
tqdm.pandas(ascii=True)
from sklearn.utils import shuffle

# shuffle 编码特征值

def reshape_feature(data_path_1,para_list_1,typepath,SEED = 888):
    data_1 = pd.read_csv(data_path_1, index_col = 'Seqname')    
    random.seed(SEED)
    # 生成所有序列对应的label name
    np.random.seed(SEED)

    data_1_T = data_1.T
    data_1_sh = shuffle(data_1_T,n_samples=data_1_T.shape[0])
    data_1_sh_T = data_1_sh.T

    data_1_np = np.concatenate((np.array(data_1_sh_T),np.zeros((data_1_sh_T.shape[0], para_list_1[0]))),axis = 1).reshape(data_1_sh_T.shape[0],para_list_1[1],para_list_1[1],1)
    np.save('/public/home/wangyx/01_MolMap/Data/CPPred/human_training_test/reshape_shuffle/Shuffle_1_'+ str(para_list_1[2]) + '_' + str(data_1_np.shape[0]) + '_' + str(data_1_np.shape[1]) + '_' + str(data_1_np.shape[2]) + '_' + str(data_1_np.shape[3]) +'.npy',data_1_np)

    subtype_data = pd.read_csv(typepath)
    types = subtype_data['Subtypes'].tolist()
    feat_nms = subtype_data['IDs'].tolist()
    dict_type = dict(zip(feat_nms,types))

    feature_nms = data_1_sh_T.columns.tolist()
    fea_npy = np.concatenate((np.array(feature_nms),np.zeros((para_list_1[0], ))),axis = 0).reshape(para_list_1[1],para_list_1[1],1)
    # with tqdm(total = data_1_np.shape[0]) as pbar:
    for sample_num in tqdm(range(data_1_np.shape[0])):
        for inde, type in enumerate(set(types)):
            # print(type)
            large_masks = np.zeros((para_list_1[1],para_list_1[1],1))
            for feat_nm in feat_nms:
                if dict_type[feat_nm] == type:
                    fea_index = np.argwhere(fea_npy== feat_nm)
                    indexs = np.squeeze(fea_index,axis = None).tolist()
                    large_masks[indexs[0], indexs[1],  indexs[2]] = data_1_np[sample_num][indexs[0], indexs[1],  indexs[2]]
            if inde == 0:
                npy_all = large_masks
            else:
                npy_all = np.concatenate((npy_all,large_masks),axis = 2)
        npy_sample = np.expand_dims(npy_all,axis = 0)
        if sample_num == 0:
            sample_all = npy_sample
        else:
            sample_all = np.concatenate((sample_all,npy_sample),axis = 0)
    np.save('/public/home/wangyx/01_MolMap/Data/CPPred/human_training_test/Shuffle_multichannel/Shuffle_multi_'+ para_list_1[2] + '_' + str(sample_all.shape[0]) + '_' + str(sample_all.shape[1]) + '_' + str(sample_all.shape[2]) + '_' + str(sample_all.shape[3]) +'.npy',sample_all)

def main():
    # descriptors ---------------------------
    data_path_0 = '/public/home/wangyx/01_MolMap/Data/CPPred/human_training_test/Descriptors_16768_772.csv'
    para_list_0 = [13,28, 'd'] #d
    typepath_0 = '/public/home/wangyx/01_MolMap/Data/RNAmap_Descriptor/subtypes.csv'
    # gaps ---------------------------
    data_path_1 = '/public/home/wangyx/01_MolMap/Data/CPPred/human_training_test/gaps_56478_725.csv'
    para_list_1 = [5,27, 'gaps']  #gap
    typepath = '/public/home/wangyx/01_MolMap/Data/RNAmap_gap/subtypes.csv'
    # gaps ---------------------------
    # reshape_feature(data_path_1,para_list_1,typepath,SEED = 888)
    reshape_feature(data_path_0,para_list_0,typepath_0,SEED = 888)

if __name__ == '__main__':
    main()