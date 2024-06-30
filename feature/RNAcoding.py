#!/usr/bin/env Python
# coding=utf-8
import os
import os.path
import sys
import pandas as pd
import numpy as np
import itertools
import Bio.SeqIO as Seq
import argparse as agp
import time
from tqdm import tqdm
tqdm.pandas(ascii=True)

### multiprocess
from multiprocessing import Process, cpu_count, Pool

sys.path.append("..")

import methods.Methods_all_standlone_molmap as Methods_all

############## function part ################################
def press_oneRNA(methodsAs, Afilepath):
    
    # allmethod = len(methodsAs) * 2
    if len(methodsAs) == 1:
        for method in methodsAs:
            Aresult = Methods_all.switch_meth(Methods_all.dictMe[method], Afilepath)
            Aresult = round(Aresult.iloc[:, :], 6)

    else:
        methodsA1 = methodsAs[0]
        Aresult = Methods_all.switch_meth(Methods_all.dictMe[methodsA1], Afilepath)
        Aresult = round(Aresult.iloc[:, :], 6)
        for i in range(1, len(methodsAs)):
            result_n = Methods_all.switch_meth(Methods_all.dictMe[methodsAs[i]], Afilepath)
            Aresult = pd.concat([Aresult, result_n], axis=1, join='inner')
            Aresult = round(Aresult.iloc[:, :], 6)
    return Aresult

def remove_uncorrectchacter(infasta):
    seqnames = []
    sequences = []
    dict_data = {}
    for seq in Seq.parse(infasta, 'fasta'):
        seqid = seq.id
        seqid = '>' + seqid
        n = 0

        sequence_o = str(seq.seq)
        sequence_o = sequence_o.upper()
        sequence_r = sequence_o.replace("U", "T")
        charterlist = ['B',  'D', 'E', 'F',  'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',  'U', 'V', 'W', 'X', 'Y', 'Z']
        for unchar in charterlist:
            # print(unchar)
            # print(type(unchar))
            sequence_r = sequence_r.replace(unchar, "")
        seqnames.append(seqid)
        sequences.append(sequence_r)
        dict_data[seqid] = sequence_r

    
    seq_names = []
    # rewrite the new fasta without wrong character
    A_save = open(infasta, 'w')
    for seqname in list(dict_data.keys()):
        a_seq = dict_data[seqname]
        if a_seq != '':
            seq_names.append(seqname)
            A_save.write(str(seqname) + '\n' + str(a_seq) + '\n')
    A_save.close()
    print('The {} is: {}'.format(infasta,len(seq_names)))
    
# RNA coding analysis---------------------------------------------------------------------------

def RNA_coding_part(Afastapath,Interfilepath,Resultpath,dimension,savetype,n_select =None):
    ######### get the processing methods ------------------------------------------------------------------------------
    # 14 classes methods
    methodsAs_1D  = ['Open reading frame (1D)', 'Transcript related (1D)', 'Codon related (1D)', 'Pseudo protein related (1D)', 'Guanine-cytosine related (1D)', 'Nucleotide related (1D)', 'Secondary structure (1D)', 'EIIP based spectrum (1D)', 'Solubility lipoaffinity (1D)', 'Partition coefficient (1D)', 'Polarizability refractivity (1D)', 'Hydrogen bond related (1D)', 'Topological indice (1D)', 'Molecular fingerprint (1D)']
    # Secondary structure (1D) 
    # methodsAs_1D  = ['Secondary structure (1D)']
    # 13 classes methods except Secondary structure (1D)
    # methodsAs_1D  = ['Open reading frame (1D)', 'Transcript related (1D)', 'Codon related (1D)', 'Pseudo protein related (1D)', 'Guanine-cytosine related (1D)', 'Nucleotide related (1D)', 'EIIP based spectrum (1D)', 'Solubility lipoaffinity (1D)', 'Partition coefficient (1D)', 'Polarizability refractivity (1D)', 'Hydrogen bond related (1D)', 'Topological indice (1D)', 'Molecular fingerprint (1D)']

    Resultfolder = os.path.exists(Resultpath)
    if not Resultfolder:
        os.makedirs(Resultpath)
 
    remove_uncorrectchacter(Afastapath)
    print('remove_uncorrectchacter is right')

    # RNA
    for n in range(1, 2, 1):
        methodsAs_1Ds = list(itertools.combinations(methodsAs_1D, n))
        
        if n_select:
            n_select = int(n_select)
            n_len_be = n_select-1
            n_len_af = n_select
        else:
            n_len_be = 0
            n_len_af = len(methodsAs_1Ds)   
        for num in range(n_len_be, n_len_af, 1):
            methodsAs_1D01 = methodsAs_1Ds[num]
            print('The {} method is starting: {}'.format(num,methodsAs_1Ds[num]))
            methodsAs_1D02 = list(methodsAs_1D01)
            if dimension == '1':
                #### coding processing
                Aresult_F_1D = press_oneRNA(methodsAs_1D02, Afastapath)
                
                # print('Aresult_F_1D.head()')
                # print(Aresult_F_1D.head())
                trainval_seq_data = pd.read_csv(Interfilepath)
                trainval_seq_data01  = trainval_seq_data.replace('>', '',regex =True)
                tranval_A = trainval_seq_data01['Seqname']
                # print('tranval_A')
                # print(tranval_A)

                A_fea = pd.merge(tranval_A, Aresult_F_1D, left_on='Seqname', right_index=True, how='left', sort=False)
                # replace NA with 0
                A_fea = A_fea.fillna(0)
                # print('A_fea.head()')
                # print(A_fea.head())
                A_res = np.array(A_fea.iloc[:,1:], np.float64)

                print('Encoding result shape')
                print(A_res.shape)

                methodsAs_1D01_01 = str(methodsAs_1D01).split("'")[1]
                if 'npy' in savetype:
                    FilepathA = os.path.join(Resultpath, str(methodsAs_1D01_01) + '.npy')
                    np.save(FilepathA, A_res)
                if 'csv' in savetype:
                    A_fea.to_csv(os.path.join(Resultpath, str(methodsAs_1D01_01) + '.csv'))
                print('The %s method is ending!' % num)

# sum the 14 classes methods into a summary file
def sum_coding(filepath,file_num):
    methodnames = ['Open reading frame (1D)', 'Transcript related (1D)', 'Codon related (1D)', 'Pseudo protein related (1D)', 'Guanine-cytosine related (1D)', 'Nucleotide related (1D)', 'Secondary structure (1D)', 'EIIP based spectrum (1D)', 'Solubility lipoaffinity (1D)', 'Partition coefficient (1D)', 'Polarizability refractivity (1D)', 'Hydrogen bond related (1D)', 'Topological indice (1D)', 'Molecular fingerprint (1D)']
    for index_m, methodname in enumerate(methodnames):
        listss = list(range(0,file_num,1))
        for index, lists in enumerate(listss):
            codingfile = os.path.join(filepath, str(lists), methodname + '.csv')
            codingdata = pd.read_csv(codingfile)
            codingdata = codingdata.iloc[:,1:]     
            if index == 0:
                codingdatas = codingdata
            else:
                codingdatas = pd.concat([codingdatas,codingdata],axis = 0)
        # save one coding method of all samples
        codingdatas.to_csv(os.path.join(filepath,methodname + '.csv'),index = False)
        print('{}. {}: {}'.format(index_m, methodname,codingdatas.shape))

        if index_m == 0:
            codingdata_alls = codingdatas
        else:
            codingdatas01 = codingdatas.iloc[:,1:]
            codingdata_alls = pd.concat([codingdata_alls,codingdatas01], axis = 1)
    # save all coding method of all samples
    # codingdata_path = os.path.join(filepath,'coding_feature_'+str(codingdata_alls.shape[0])+'_'+str(codingdata_alls.shape[1])+'.csv')
    filepath01 = filepath[:-1]
    codingdata_path = filepath01 + '_descriptor_'+str(codingdata_alls.shape[0])+'_'+str(codingdata_alls.shape[1])+'.csv'

    codingdata_alls.to_csv(codingdata_path,index = False)

    print('Final encoding result: {}'.format(codingdata_alls.shape))

    return codingdata_path

# split the fasta into num small files for encoding processing
def split_fasta(fasta,resilt_path,paralle = 10):

    os.makedirs(resilt_path, exist_ok=True)
    seqname = []
    for seq in Seq.parse(fasta,'fasta'):
        seqname.append(seq.id)
    # print(len(seqname))
    
    
    step = int(len(seqname)/paralle)
    all_seqs = [seqname[i:i+step] for i in range(0, len(seqname), step)]

    for seq in Seq.parse(fasta,'fasta'):
        for index in range(len(all_seqs)):

            if seq.id in all_seqs[index]:
                path =  resilt_path + str(index) +'.fa'
                with open(path, 'a+') as f:
                    seqid = '>' + seq.id
                    f.write(seqid)
                    f.write('\n')
                    f.write(str(seq.seq))
                    f.write('\n')

    for index in range(len(all_seqs)):
        csvpath = resilt_path + str(index) +'.csv'
        names = list(map(lambda x: '>' + x, all_seqs[index]))
        labels = [1 for x in range(len(names))]
        dict_name = {
            'Seqname': names,
            'Label': labels
             }
        df = pd.DataFrame(dict_name)
        df.to_csv(csvpath, index = None)

    if len(seqname)%paralle == 0:
        filenum = paralle
    else:
        filenum = paralle + 1
    print('Split fasta with the {} sequence into {} files'.format(len(seqname), filenum))
    return filenum


def RNA_coding_descriptors(fasta_path,output_path):   

    filepath_01 = fasta_path.split('/')[-1].split('.')[0]


    # First step: split the fasta into small files--------------------------------------------------

    resilt_path = output_path + '/'+ filepath_01 + '/'
    filenum = split_fasta(fasta_path,resilt_path,paralle = 4)
    
    # Second step: encoding features parallelly------------------------------------------------------
    params = []
    # listss = list(range(0,filenum,1))
    for index in tqdm(range(0,filenum,1), ascii=True):
        Afastapath = os.path.join(resilt_path, str(index) + '.fa')
        Interfilepath = os.path.join(resilt_path, str(index) + '.csv')
        Resultpath = os.path.join(output_path, resilt_path, str(index))
        savetype = 'npycsv'
        dimension = '1'
        
        onepara = (Afastapath,Interfilepath,Resultpath,dimension, savetype)
        params.append(onepara)

    start_time = time.time()
	### multiprocess pool
    p = Pool(filenum)
    for param in params:
        print(param)
        p.apply_async(RNA_coding_part, args = param)
    p.close()
    p.join()
    end_time = time.time()
    print('Total cost the time is {} minutes'.format((end_time - start_time)/60))
    
    # Third: sum all the encoding features----------------------------------------------------------
    codingdata_alls = sum_coding(resilt_path,filenum)

    return codingdata_alls


