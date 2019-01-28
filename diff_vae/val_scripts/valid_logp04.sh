#!/bin/bash

DIR=$1
NUM=$2

for ((i=1; i<NUM; i++)); do
	f=$DIR/model.iter-$i
	echo $f
	python decode.py --test ../data/logp04/valid.txt --vocab ../data/logp04/vocab.txt --model $f | python ../scripts/logp_score.py > $DIR/results.$i
	python ../scripts/logp_analyze.py --delta 0.4 < $DIR/results.$i
done
