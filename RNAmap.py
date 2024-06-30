#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Wangyunxia, Date: 20220413
import os
import pandas as pd

from map.RNA_featurizer_map_d import transform_d
from map.RNA_featurizer_map_gap import transform_gap

# First step: ==============================================================
# encoding fasta into encoding feature
# make transform the encoding feature into map file
# ===========================================================================

map_type = '5species'# ['human','5species']

fastapath = os.path.abspath(os.path.dirname(__file__)) + '/demo/sense_intronic_sample.fa'
# fastapath = os.path.abspath(os.path.dirname(__file__)) + '/demo/lncRNA_xist.fa'
# fastapath = '/public/home/wangyx/01_MolMap/Data/fasta_data/ENCODE/1000.fa'
# fastapath = '/public/home/wangyx/01_MolMap/Data/fasta_data/ENCODE/4_test.fa'

output_path = os.path.abspath(os.path.dirname(__file__)) + '/demo/output_sense_intronic_sample'
# output_path = '/public/home/wangyx/01_MolMap/Data/fasta_data/ENCODE/out'

feature_d = transform_d(fastapath,map_type,output_path)
feature_gap = transform_gap(fastapath,map_type,output_path)

print(feature_d.shape)
print(feature_d)
print(feature_gap.shape)
print(feature_gap)

# Second step: =================================================================================================
# training the model
# =================================================================================================

# load the transformed Y label
y_train_valid_path = os.path.abspath(os.path.dirname(__file__)) + '/demo/sense_intronic_sample_label.csv'













