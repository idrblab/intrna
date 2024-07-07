#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Wangyunxia, Date: 20240707
import os
import pandas as pd
import numpy as np
import argparse as agp
import tensorflow as tf
tf.random.set_seed(888)


from map.RNA_featurizer_map_d import transform_d
from map.RNA_featurizer_map_gap import transform_gap



def predict_model(fastapath,output_path,model_number):

    # First step: ==============================================================
    # encoding fasta into encoding feature
    # make transform the encoding feature into map file
    # ===========================================================================
    map_type = '5species'
    feature_d = transform_d(fastapath,map_type,output_path)
    feature_gap = transform_gap(fastapath,map_type,output_path)

    print(feature_d.shape)
    print(feature_d)
    print(feature_gap.shape)
    print(feature_gap)

    # Second step: =================================================================================================
    # training the model
    # =================================================================================================

    testX = (feature_d, feature_gap)

    if model_number == 1:
        # ---------------------------------------
        # classify the RNA into mRNA or ncRNA
        model_path = os.path.abspath(os.path.dirname(__file__)) + '/model/trained_model/five_species_mRNA_ncRNA_model'
        my_model = tf.keras.models.load_model(model_path)
        y_proba = my_model.predict(testX)
        y_group = np.round(y_proba)

        y_proba_df = pd.DataFrame(y_proba)
        y_group_df = pd.DataFrame(y_group)
        y_group_df.to_csv(output_path + '/classified_result_for_mRNA_ncRNA.csv')

    elif model_number == 2:

        # ---------------------------------------
        # classify the RNA into 13 classes ncRNA
        model_path = os.path.abspath(os.path.dirname(__file__)) + '/model/trained_model/five_species_13_ncRNA_model'
        my_model = tf.keras.models.load_model(model_path)
        y_proba = my_model.predict(testX)
        y_group = np.round(y_proba)

        y_proba_df = pd.DataFrame(y_proba)
        y_group_df = pd.DataFrame(y_group)
        y_group_df.to_csv(output_path + '/classified_result_for_13classes_ncRNA.csv')

    elif model_number == 3:
        # ---------------------------------------
        # classify the RNA into linearRNA and circRNA
        model_path = os.path.abspath(os.path.dirname(__file__)) + '/model/trained_model/five_species_circRNA_linearRNA_model'
        my_model = tf.keras.models.load_model(model_path)
        y_proba = my_model.predict(testX)
        y_group = np.round(y_proba)

        y_proba_df = pd.DataFrame(y_proba)
        y_group_df = pd.DataFrame(y_group)
        y_group_df.to_csv(output_path + '/classified_result_for_circRNA_linearRNA.csv')
    else:
        print('Model number is wrong! Please input the correct model number!')

def main():

     # get parameters from out input
    parser = agp.ArgumentParser(description="Your program description here")
    parser.add_argument('-i','--inputfasta', help='The RNA data needed to predict')
    parser.add_argument('-o','--outputpath', help='The folder to save encoding features and predict result')
    parser.add_argument('-m','--model_type', default=1, choices=[1, 2, 3],help='The model will be used to predict. 1: mRNA and ncRNA; 2: 13 classes linear ncRNA; 3: circRNA and linear RNA')
    args = parser.parse_args()

    # fastapath = os.path.abspath(os.path.dirname(__file__)) + '/demo/sense_intronic_sample.fa'
    # output_path = os.path.abspath(os.path.dirname(__file__)) + '/demo/output_sense_intronic_sample'
    # model_number = 1  

    fastapath = args.inputfasta
    output_path = args.outputpath
    model_number = args.model_type     

    predict_model(fastapath,output_path,model_number)


if __name__=='__main__':
    main()
