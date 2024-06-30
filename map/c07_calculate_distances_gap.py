import sys

from utils import distances, calculator
import feature
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(ascii=True)


def loadnpy(filename, N, dtype, mode = 'r'):
    f = np.memmap(filename, mode = mode, 
                  dtype = dtype)
    M = int(len(f) / N)
    print(M, N)
    f = f.reshape(M, N)
    return f


def caldis(savepath, data, idx, tag, methods = ['cosine']):
    
    
    ##############################################################
    ## to test one can use smaller number of compounds ##
    #############################################################
    
    
    for method in methods:
        res = calculator.pairwise_distance(data, n_cpus=12, method=method)
        
        res = np.nan_to_num(res,copy=False)
        df = pd.DataFrame(res,index=idx,columns=idx)
        df.to_csv(savepath + '%s_%s.csv' % (tag, method))
        df.to_pickle(savepath + '%s_%s.cfg' % (tag, method))


def calculate_distance_gap(datapath,savepath):
    # data_nm = pd.read_csv('/public/home/wangyx/01_MolMap/Data/RNAmap_gap/gaps_417909_724.csv')
    data_nm = pd.read_csv(datapath)
    data = np.array(data_nm.iloc[:,1:])
    # 特征归一化到[0,1]的范围
    min_max_scaler = MinMaxScaler()
    Ddata = min_max_scaler.fit_transform(data)#归一化后的结果
    idx = data_nm.columns[1:]
    tag = 'descriptor'
    caldis(savepath, Ddata, idx, tag, methods = ['cosine'])


# if __name__ == '__main__':
    
    #discriptors distance
    # Nd = len(feature.descriptor.Extraction().bitsinfo)
    # Nd = 561
    # idx = feature.descriptor.Extraction().bitsinfo.IDs.tolist()
    # data = loadnpy('./data/descriptors_8348036.npy', N = Nd, dtype = np.float)
    
    # data_nm = pd.read_csv('/public/home/wangyx/01_MolMap/Data/RNAmap_gap/gaps_417909_724.csv')
    # data = np.array(data_nm.iloc[:,1:])
    # # 特征归一化到[0,1]的范围
    # min_max_scaler = MinMaxScaler()
    # Ddata = min_max_scaler.fit_transform(data)#归一化后的结果
    # idx = data_nm.columns[1:]
    # tag = 'descriptor'
    # caldis(Ddata, idx, tag, methods = ['cosine'])
    
