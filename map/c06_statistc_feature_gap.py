
# from molmap import feature
from utils import summary
# import sys
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
# from rdkit import Chem
from tqdm import tqdm
tqdm.pandas(ascii=True)

def savenpy(filename, data):
    f = np.memmap(filename, mode = 'w+', 
                  shape = data.shape, 
                  dtype = data.dtype)
    f[:] = data
    f.flush()
    del f
    

def loadnpy(filename, N, dtype):
    f = np.memmap(filename, mode = 'r', 
                  dtype = dtype)
    M = int(len(f) / N)
    print(M, N)
    f = f.reshape(M, N)
    return f

# Nd = len(feature.descriptor.Extraction().bitsinfo)
# Nd = 561
def statistic_feature_gap(datapath,savepath):
    S = summary.Summary(n_jobs = 10)

    # datapath = '/public/home/wangyx/01_MolMap/Data/RNAmap_gap'
    # Ddata = loadnpy('./data/descriptors_8348036.npy', N = Nd, dtype = np.float)
    # Ddata = loadnpy('./data/RNA_descrip.npy', N = Nd, dtype = np.float)
    # data_nm = pd.read_csv(os.path.join(datapath,'gaps_417909_724.csv'))
    data_nm = pd.read_csv(datapath)
    Ddata = np.array(data_nm.iloc[:,1:])
    # 特征归一化到[0,1]的范围
    min_max_scaler = MinMaxScaler()
    Ddata = min_max_scaler.fit_transform(Ddata)#归一化后的结果
    # Ddata = np.load('./data/RNA_descrip.npy')
    print('Ddata')
    print(Ddata.shape)
    res= []
    for i in tqdm(range(Ddata.shape[1])):
        r = S._statistics_one(Ddata, i)
        res.append(r)
        
    df = pd.DataFrame(res)
    # print('feature.descriptor.Extraction().bitsinfo.IDs')
    # print(feature.descriptor.Extraction().bitsinfo.IDs)

    colnam = data_nm.columns[1:]
    # print('colnam')
    # print(colnam)
    # df.index = feature.descriptor.Extraction().bitsinfo.IDs
    df.index = colnam
    print(df)
    df.to_csv(os.path.join(savepath,'descriptor_scale.csv'))
    df.to_pickle(os.path.join(savepath,'descriptor_scale.cfg'))


