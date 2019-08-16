#!/bin/bash
analysis_num=$1
dataset='./datasets/spoken_arabic_digits/train'
data_size=13
num_derivative=2
n_iter=10
states_num=5
GMM_mix_num=3
covariance_type='diag'

python hmm_train.py --analysis $analysis_num --dataset $dataset --data_size $data_size --num_derivative $num_derivative --n_iter $n_iter --states_num $states_num --gmm_mix_num $gmm_mix_num --covariance_type $covariance_type