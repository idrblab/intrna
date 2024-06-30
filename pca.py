#coding=gbk
import numpy as np
from sklearn.decomposition import PCA
from Bio import SeqIO
# x = np.load(r"E:\xiawq\encoding\new_encoding\rna_oneht.npy")
# x = x.reshape(5, 8000)
# # X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #�������ݣ�ά��Ϊ4
# pca = PCA(n_components=5)   #����2ά
# pca.fit(x)                  #ѵ��
# newX=pca.fit_transform(x)   #��ά�������
# # PCA(copy=True, n_components=2, whiten=False)
# print(pca.explained_variance_ratio_)  #���������
# print(newX)

def reduce_pca(file):

    # encoded = np.load(file)
    encoded = file

    n = encoded.shape[0]

    encoded = encoded.reshape(n, -1)
    if n >= 256:
        n_components = 256
    if n < 256:
        n_components = n
    pca = PCA(n_components)
    pca.fit(encoded)
    newX = pca.fit_transform(encoded)
    return newX
# if __name__ == '__main__':
#     newX = reduce_pca(r"E:\xiawq\encoding\new_encoding\rna_oneht.npy")
#     print(newX)

