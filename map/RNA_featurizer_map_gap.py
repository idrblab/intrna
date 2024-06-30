#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("..")

from utils.logtools import print_info, print_error
from utils.matrixopt import Scatter2Grid, Scatter2Array
from utils import vismap
from feature.gap_feature import calculate_gap_features

from sklearn.manifold import TSNE, MDS
from joblib import Parallel, delayed, load, dump
from umap import UMAP
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
tqdm.pandas(ascii=True)

class Base:
    
    def __init__(self):
        pass
        
    def _save(self, filename):
        return dump(self, filename)
        
    def _load(self, filename):
        return load(filename)
 

    def MinMaxScaleClip(self, x, xmin, xmax):
        scaled = (x - xmin) / ((xmax - xmin) + 1e-8)
        return scaled.clip(0, 1)

    def StandardScaler(self, x, xmean, xstd):
        return (x-xmean) / (xstd + 1e-8) 
    
    
    
class MolMap(Base):
    
    def __init__(self,
                 save_path, 
                 ftype = 'descriptor',
                 flist = None, 
                 fmap_type = 'grid', 
                 fmap_shape = None, 
                 split_channels = True,
                 metric = 'cosine', 
                 var_thr = 1e-4, ):
        """
        paramters
        -----------------
        ftype: {'fingerprint', 'descriptor'}, feature type
        flist: feature list, if you want use some of the features instead of all features, each element in flist should be the id of a feature
        fmap_shape: None or tuple, size of molmap, only works when fmap_type is 'scatter', if None, the size of feature map will be calculated automatically
        fmap_type:{'scatter', 'grid'}, default: 'gird', if 'scatter', will return a scatter mol map without an assignment to a grid
        split_channels: bool, if True, outputs will split into various channels using the types of feature
        metric: {'cosine', 'correlation'}, default: 'cosine', measurement of feature distance
        var_thr: float, defalt is 1e-4, meaning that feature will be included only if the conresponding variance larger than this value. Since some of the feature has pretty low variances, we can remove them by increasing this threshold
        """
        
        super().__init__()
        assert ftype in ['descriptor', 'fingerprint'], 'no such feature type supported!'        
        assert fmap_type in ['scatter', 'grid'], 'no such feature map type supported!'
       
        self.ftype = ftype
        self.metric = metric
        self.method = None
        self.isfit = False

        
        #default we will load the  precomputed matrix
        # dist_matrix = load_config(ftype, metric)
        # dist_matrix = load_config(ftype, metric)
        # dist_matrix = pd.read_csv('/public/home/wangyx/01_MolMap/Data/RNAmap_gap/descriptor_cosine.csv')
        dist_matrix = pd.read_csv(save_path + '/descriptor_cosine.csv')
        dist_matrix.index = dist_matrix.iloc[:,0].tolist()
        dist_matrix = dist_matrix.iloc[:,1:]
        # print('dist_matrix')
        # print(dist_matrix)


        feature_order = dist_matrix.index.tolist()
        feat_seq_dict = dict(zip(feature_order, range(len(feature_order))))


        # scale_info = load_config(ftype, 'scale')
        # scale_info = pd.read_csv('/public/home/wangyx/01_MolMap/Data/RNAmap_gap/descriptor_scale_417909_724.csv')      
        scale_info = pd.read_csv(save_path + '/descriptor_scale.csv')
        scale_info.index = scale_info.iloc[:,0].tolist()

        scale_info = scale_info.iloc[:,1:]
        # print('scale_info')
        # print(scale_info)
        scale_info = scale_info[scale_info['var'] > var_thr]

        slist = scale_info.index.tolist()
        
        if not flist:
            flist = list(dist_matrix.columns)
        
        #fix input feature's order as random order
        final_list = list(set(slist) & set(flist))
        final_list.sort(key = lambda x:feat_seq_dict.get(x))
        #final_list = shuffle(final_list, random_state=123)

        dist_matrix = dist_matrix.loc[final_list][final_list]
        
        self.dist_matrix = dist_matrix
        self.flist = final_list
        self.scale_info = scale_info.loc[final_list]
        
        #init the feature extract object
        # if ftype == 'fingerprint':
        #     self.extract = fext()
        # else:
        #     self.extract = dext() 

        self.colormaps = {
                'AA': '#d62728',
                'TT': '#ff9896',
                'AT': '#e377c2',
                'TA': '#f7b6d2',

                'AG': '#ff7f0e',
                'GA': '#ffbb78',
                'TC': '#bcbd22',
                'CT': '#dbdb8d',
                'AC': '#2ca02c',
                'CA': '#98df8a',
                'GT': '#17becf',
                'TG': '#9edae5',

                'CC': '#1f77b4',
                'GG': '#aec7e8',
                'GC': '#9467bd',
                'CG': '#c5b0d5',

                'kmer': '#8c564b',          
                
                'NaN': '#000000'
            }
        
        # df = pd.DataFrame(arr).T
        # bitsinfo = pd.read_csv('/public/home/wangyx/01_MolMap/Data/RNAmap_gap/subtypes.csv')
        bitsinfo = pd.read_csv(save_path + '/subtypes.csv')
        
        bitsinfo['colors'] = bitsinfo.Subtypes.map(self.colormaps)
        self.bitsinfo = bitsinfo
        self.fmap_type = fmap_type
        
        if fmap_type == 'grid':
            S = Scatter2Grid()
        else:
            if fmap_shape == None:
                N = len(self.flist)
                l = np.int(np.sqrt(N))*2
                fmap_shape = (l, l)                
            S = Scatter2Array(fmap_shape)
        
        self._S = S
        self.split_channels = split_channels        

        
        
    def _fit_embedding(self, 
                        method = 'tsne',  
                        n_components = 2,
                        random_state = 1,  
                        verbose = 2,
                        n_neighbors = 30,
                        min_dist = 0.1,
                        **kwargs):
        
        """
        parameters
        -----------------
        method: {'tsne', 'umap', 'mds'}, algorithm to embedd high-D to 2D
        kwargs: the extra parameters for the conresponding algorithm
        """
        dist_matrix = self.dist_matrix
        if 'metric' in kwargs.keys():
            metric = kwargs.get('metric')
            kwargs.pop('metric')
            
        else:
            metric = 'precomputed'

        if method == 'tsne':
            embedded = TSNE(n_components=n_components, 
                            random_state=random_state,
                            metric = metric,
                            verbose = verbose,
                            **kwargs)
        elif method == 'umap':
            embedded = UMAP(n_components = n_components, 
                            n_neighbors = n_neighbors,
                            min_dist = min_dist,
                            verbose = verbose,
                            random_state=random_state, 
                            metric = metric, **kwargs)
            
        elif method =='mds':
            if 'metric' in kwargs.keys():
                kwargs.pop('metric')
            if 'dissimilarity' in kwargs.keys():
                dissimilarity = kwargs.get('dissimilarity')
                kwargs.pop('dissimilarity')
            else:
                dissimilarity = 'precomputed'
                
            embedded = MDS(metric = True, 
                           n_components= n_components,
                           verbose = verbose,
                           dissimilarity = dissimilarity, 
                           random_state = random_state, **kwargs)

        embedded = embedded.fit(dist_matrix)    

        df = pd.DataFrame(embedded.embedding_, index = self.flist,columns=['x', 'y'])
        typemap = self.bitsinfo.set_index('IDs')
        
        df = df.join(typemap)
        df['Channels'] = df['Subtypes']
        print('df')
        print(df)
        self.df_embedding = df
        self.embedded = embedded
        # print('df')
        # print(df)
        

    def fit(self, 
            method = 'umap', min_dist = 0.1, n_neighbors = 30,
            verbose = 2, random_state = 1, **kwargs): 
        """
        parameters
        -----------------
        method: {'tsne', 'umap', 'mds'}, algorithm to embedd high-D to 2D
        kwargs: the extra parameters for the conresponding method
        """
        if 'n_components' in kwargs.keys():
            kwargs.pop('n_components')
            
        ## embedding  into a 2d 
        assert method in ['tsne', 'umap', 'mds'], 'no support such method!'
        
        self.method = method
        
        ## 2d embedding first
        self._fit_embedding(method = method,
                            n_neighbors = n_neighbors,
                            random_state = random_state,
                            min_dist = min_dist, 
                            verbose = verbose,
                            n_components = 2, **kwargs)

        
        if self.fmap_type == 'scatter':
            ## naive scatter algorithm
            print_info('Applying naive scatter feature map...')
            self._S.fit(self.df_embedding, self.split_channels, channel_col = 'Channels')
            print_info('Finished')
            
        else:
            ## linear assignment algorithm 
            print_info('Applying grid feature map(assignment), this may take several minutes(1~30 min)')
            self._S.fit(self.df_embedding, self.split_channels, channel_col = 'Channels')
            print_info('Finished')
        
        ## fit flag
        self.isfit = True
        self.fmap_shape = self._S.fmap_shape
        print(self.fmap_shape)
        # print(self.fmap_shape)
    

    
    def transform(self, 
                  df,
                  scale = True, 
                  scale_method = 'minmax',):
    
    
        """
        parameters
        --------------------
        smiles:smiles string of compound
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}
        """
        
        if not self.isfit:
            print_error('please fit first!')
            return

        # arr = self.extract.transform(smiles)



        df.columns = self.bitsinfo.IDs
        
        if (scale) & (self.ftype == 'descriptor'):
            
            if scale_method == 'standard':
                df = self.StandardScaler(df,  
                                    self.scale_info['mean'],
                                    self.scale_info['std'])
            else:
                df = self.MinMaxScaleClip(df, 
                                     self.scale_info['min'], 
                                     self.scale_info['max'])
        
        df = df[self.flist]
        vector_1d = df.values[0] #shape = (N, )
        fmap = self._S.transform(vector_1d)       
        return np.nan_to_num(fmap)   
        

        
    def batch_transform(self, 
                        smiles_list, 
                        scale = True, 
                        scale_method = 'minmax',
                        n_jobs=4):
    
        """
        parameters
        --------------------
        smiles_list: list of smiles strings
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}
        n_jobs: number of parallel
        """
        
                    
        P = Parallel(n_jobs=n_jobs)
        res = P(delayed(self.transform)(smiles, 
                                        scale,
                                        scale_method) for smiles in tqdm(smiles_list, ascii=True)) 
        X = np.stack(res) 
        
        return X

    
    def rearrangement(self, orignal_X, target_mp):

        """
        Re-Arragement feature maps X from orignal_mp's to target_mp's style, in case that feature already extracted but the position need to be refit and rearrangement.

        parameters
        -------------------
        orignal_X: the feature values transformed from orignal_mp(object self)
        target_mp: the target feature map object

        return
        -------------
        target_X, shape is (N, W, H, C)
        """
        assert self.flist == target_mp.flist, print_error('Input features list is different, can not re-arrangement, check your flist by mp.flist method' )
        assert len(orignal_X.shape) == 4, print_error('Input X has error shape, please reshape to (samples, w, h, channels)')
        
        idx = self._S.df.sort_values('indices').idx.tolist()
        idx = np.argsort(idx)

        N = len(orignal_X) #number of sample
        M = len(self.flist) # number of features
        res = []
        for i in tqdm(range(N), ascii=True):
            x = orignal_X[i].sum(axis=-1)
            vector_1d_ordered = x.reshape(-1,)
            vector_1d_ordered = vector_1d_ordered[:M]
            vector_1d = vector_1d_ordered[idx]
            fmap = target_mp._S.transform(vector_1d)
            res.append(fmap)
        return np.stack(res)

    
    
    def plot_scatter(self, htmlpath='./', htmlname=None, radius = 3):
        """radius: the size of the scatter, must be int"""
        df_scatter, H_scatter = vismap.plot_scatter(self,  
                                htmlpath=htmlpath, 
                                htmlname=htmlname,
                                radius = radius)
        
        self.df_scatter = df_scatter
        return H_scatter   
        
        
    def plot_grid(self, htmlpath='./', htmlname=None):
        
        if self.fmap_type != 'grid':
            return
        
        df_grid, H_grid = vismap.plot_grid(self,  
                                htmlpath=htmlpath, 
                                htmlname=htmlname)
        
        self.df_grid = df_grid
        return H_grid       
        
        
    def load(self, filename):
        return self._load(filename)
    
    
    def save(self, filename):
        return self._save(filename)


def get_color_dict(mp):
    df = mp._S.df
    return df.set_index('Subtypes')['colors'].to_dict()


def show_fmap(mp, X, figsize=(6,6), fname = './1.pdf'):
    
    mp_colors = get_color_dict(mp)
    fig =  plt.figure(figsize=figsize)
    channels = [i for i in mp.colormaps.keys() if i in mp._S.channels ]
             
    
    for i, j  in enumerate(channels):

        data = X[:,:,mp._S.channels.index(j)]
        color = mp_colors[j]
        if mp.ftype == 'fingerprint':
            # print('mp.ftype: {}'.format(mp.ftype))
            # cmap = sns.dark_palette(color, n_colors =  2, reverse=True)
            cmap = sns.light_palette(color, n_colors =  100, reverse=False)
        else:
            cmap = sns.light_palette(color, n_colors =  100, reverse=False)
        
        ax = sns.heatmap(np.where(data !=0, data, np.nan), 
                    cmap = cmap, 
                    yticklabels=False, xticklabels=False, cbar=False, 
                    linewidths=0.005, linecolor = '0.9')# cbar_kws = dict(use_gridspec=False,location="top")

    ax.axhline(y=0, color='grey',lw=2, ls =  '--')
    ax.axvline(x=data.shape[1], color='grey',lw=2, ls =  '--')
    ax.axhline(y=data.shape[0], color='grey',lw=2, ls =  '--')
    ax.axvline(x=0, color='grey',lw=2, ls =  '--')

    patches = [ plt.plot([],[], marker="s", ms=8, ls="", mec=None, color=j, 
                label=i)[0]  for i,j in mp.colormaps.items() if i in channels]
    
    l = 1.32
    if mp.ftype == 'fingerprint':
        l -= 0.05
    plt.legend(handles=patches, bbox_to_anchor=(l,1.01), 
               loc='upper right', ncol=1, facecolor="w", numpoints=1 )    
    
    #plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight', dpi = 400)

def make_transform(savepath, fname,output_path):
    ######## First step start: make transformed X and Y from raw coding feature files-----------------------
    # define your molmap
    mp = MolMap(savepath,ftype = 'fingerprint', fmap_type = 'grid',
                    split_channels = True,   metric='cosine', var_thr=1e-4)
    tem_output = output_path + '/png_result_gap'
    os.makedirs(tem_output, exist_ok=True)
    # Fit your molmap
    mp.fit(method = 'umap', verbose = 2)
    mp_name = tem_output + '/RNA_features_gap.mp'
    mp.save(mp_name)
    mp.plot_grid()
    df_grid = mp.df_grid
    df_grid.to_csv(tem_output + '/df_grid_gap.csv')
    # Visulization of your molmap
    mp.plot_scatter()

    chunksize = 1024
    
    iterator = pd.read_csv(fname, 
                             iterator=True, index_col = 0,
                             chunksize= chunksize)
    
    fasta_file = pd.read_csv(fname)
    r = fasta_file.shape[0] // chunksize
    # molmap transform
    npy_all = []
    with tqdm(total = r) as pbar:
        start = 0
        for i,df in tqdm(enumerate(iterator), ascii=True):
            end = start + len(df)
            npy = []
            for j in range(len(df)):
                num = start+j
                num = str(num)
                data_tem = df.iloc[j,:].to_frame().T

                X = mp.transform(data_tem,scale = True, 
                    scale_method = 'minmax')
                # print(X.shape)
                npy.append(X)
                npy_all.append(X)

                fname= tem_output + '/' + str(df.iloc[j,:].to_frame().columns[0]) + '.png'
                show_fmap(mp, X, fname = fname )

            print(start,end)
            start = end
            pbar.update(1)
    npy_all = np.array(npy_all)
    print('npy_all.shape: {}'.format(npy_all.shape))
    np.save(output_path + '/transformed_feature_gap.npy',npy_all)
    return npy_all

def transform_gap(fastapath,map_type,output_path):

######## First step: make raw encoding feature file from fasta files-----------------------
    codingdata_path = calculate_gap_features(fastapath,output_path)
    # codingdata_path = '/public/home/wangyx/01_MolMap/code/CNN_gap/demo/output_lncRNA_xist/lncRNA_xistlncRNA_xist_gaps_4_724.csv'
######## Second step: make map file from raw encoding feature files-------------------------
    if map_type == 'human':
        # human_map
        savepath = os.path.abspath(os.path.dirname(__file__)) + '/RNAmap_gap/human/'
    elif map_type == '5species':
        # 5species_map
        savepath = os.path.abspath(os.path.dirname(__file__)) + '/RNAmap_gap/5species/'

    feature_transform_gap = make_transform(savepath,codingdata_path,output_path)            
    return feature_transform_gap
