#!/bin/bash

# logp06
mkdir -p newmodels/logp06
python vae_train.py --train ../data/logp06/processed/ --vocab ../data/logp06/vocab.txt --save_dir newmodels/logp06 --hidden_size 300 --rand_size 16 --anneal_rate 0.85 --epoch 10 --beta 2.0 --batch_size 20 --share_embedding | tee newmodels/logp06/LOG

# logp04
mkdir -p newmodels/logp04
python vae_train.py --train ../data/logp04/processed/ --vocab ../data/logp04/vocab.txt --save_dir newmodels/logp04 --hidden_size 330 --rand_size 8 --anneal_rate 0.8 --epoch 10 --share_embedding | tee newmodels/logp04/LOG

# QED 
mkdir -p newmodels/qed
python vae_train.py --train ../data/qed/processed/ --vocab ../data/qed/vocab.txt --save_dir newmodels/qed --hidden_size 300 --rand_size 8 --anneal_rate 0.8 --epoch 10 --use_molatt | tee newmodels/qed/LOG

# DRD2 
mkdir -p newmodels/drd2
python vae_train.py --train ../data/drd2/processed/ --vocab ../data/drd2/vocab.txt --save_dir newmodels/drd2 --hidden_size 300 --rand_size 8 --batch_size 20 --epoch 20 --use_molatt | tee newmodels/drd2/LOG
