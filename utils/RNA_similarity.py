from .calculator import *

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(ascii=True)
# import multiprocessing

def loadnpy(filename, N, dtype, mode = 'r'):
    f = np.memmap(filename, mode = mode, 
                  dtype = dtype)
    M = int(len(f) / N)
    print(M, N)
    f = f.reshape(M, N)
    return f


def caldis(data, idx, tag, methods = ['cosine']):
    
    
    ##############################################################
    ## to test one can use smaller number of compounds ##
    #############################################################
    
    
    for method in methods:
        # n_cpus = multiprocessing.cpu_count()
        res = pairwise_distance_row(data, n_cpus=18, method=method)
        res = np.nan_to_num(res,copy=False)
        df = pd.DataFrame(res,index=idx,columns=idx)
        # df.to_pickle('./data/%s_%s.cfg' % (tag, method))

        return df

def caldis_two(data_ref, data_get, colnum, idx, tag, methods = ['cosine']):
    
    
    ##############################################################
    ## to test one can use smaller number of compounds ##
    #############################################################
    
    
    for method in methods:
        res = pairwise_distance_two(data_ref, data_get, n_cpus=20, method=method)
        res = np.nan_to_num(res,copy=False)
        df = pd.DataFrame(res,index=idx,columns=colnum)
        # df.to_pickle('./data/%s_%s.cfg' % (tag, method))

        return df

def caldis_same(data_nm):
    data = np.array(data_nm.iloc[:,1:])
    idx = data_nm.iloc[:,0]
    tag = 'descriptor'
    res = caldis(data, idx, tag, methods = ['cosine'])
    return res

def caldis_twofiles(data_0_ref,data_1_get):
    data_0 = np.array(data_0_ref.iloc[:,1:])
    data_1 = np.array(data_1_get.iloc[:,1:])
    idx = data_1_get.iloc[:,0]
    tag = 'descriptor'
    colnum = data_0_ref.iloc[:,0]
    res = caldis_two(data_0, data_1, colnum, idx, tag, methods = ['cosine'])
    return res

def main():
    data_nm = pd.read_csv('/public/home/wangyx/01_MolMap/code/Data/sORF_data/CPPredData_test_human_sorf_D_1282.csv')
    data_1_get = pd.read_csv('/public/home/wangyx/01_MolMap/code/Data/DeepCPPdata/coding_feature/DeepCPP_human_sorf.csv')
    res = caldis_same(data_nm)
    # print(res.shape)
    # res.to_csv('/public/home/wangyx/01_MolMap/code/Data/sORF_data/CPPred_1282_row_similar.csv')
    res = caldis_twofiles(data_nm,data_1_get)
    res.to_csv('/public/home/wangyx/01_MolMap/code/Data/sORF_data/CPPred_1282_464_similar.csv')
    print(res.shape)



if __name__ == '__main__':
    main()
    
#     #discriptors distance
#     Nd = len(feature.descriptor.Extraction().bitsinfo)
#     idx = feature.descriptor.Extraction().bitsinfo.IDs.tolist()
#     data = loadnpy('./data/descriptors_8348036.npy', N = Nd, dtype = np.float)
#     tag = 'descriptor'
#     caldis(data, idx, tag, methods = ['cosine'])
    
