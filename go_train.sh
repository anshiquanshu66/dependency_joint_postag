#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python graph_parser_joint_pos.py --cuda --mode LSTM --num_epochs 1000 --batch_size 80 --hidden_size 512 --num_layers 3 \
 --pos_dim 100 --char_dim 64 --num_filters 100 --arc_space 512 --type_space 128 \
 --opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --schedule 10 --gamma 0.0 --clip 5.0 \
 --p_in 0.33 --p_rnn 0.33 0.33 --p_out 0.33 --unk_replace 0.5 --pos --char \
 --objective cross_entropy --decode greedy \
 --char_embedding random \
 --punctuation '.' '``' "''" ':' ',' \
 --train "/home/hnc/PycharmProjects/dependency-vtcc/data/ud-treebanks-v2.2/UD_Vietnamese-VTB/vi_vtb-ud-train.conllu" \
 --dev "/home/hnc/PycharmProjects/dependency-vtcc/data/ud-treebanks-v2.2/UD_Vietnamese-VTB/vi_vtb-ud-dev.conllu" \
 --test "/home/hnc/PycharmProjects/dependency-vtcc/data/ud-treebanks-v2.2/UD_Vietnamese-VTB/vi_vtb-ud-test.conllu" \
 --model_path "results/biaffine_pos/" --model_name 'network.pt'
