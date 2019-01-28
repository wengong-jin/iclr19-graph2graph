#!/bin/bash

python ../scripts/preprocess.py --train ../data/logp06/train_pairs.txt
mkdir processed
mv tensor* processed
mv processed ../data/logp06

python ../scripts/preprocess.py --train ../data/logp04/train_pairs.txt
mkdir processed
mv tensor* processed
mv processed ../data/logp04

python ../scripts/preprocess.py --train ../data/qed/train_pairs.txt
mkdir processed
mv tensor* processed
mv processed ../data/qed

python ../scripts/preprocess.py --train ../data/drd2/train_pairs.txt
mkdir processed
mv tensor* processed
mv processed ../data/drd2
