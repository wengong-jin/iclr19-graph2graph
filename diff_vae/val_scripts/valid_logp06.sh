#!/bin/bash

DIR=$1
NUM=$2

for ((i=1; i<NUM; i++)); do
	f=$DIR/model.iter-$i
	echo $f
	python decode.py --test ../data/logp06/valid.txt --vocab ../data/logp06/vocab.txt --model $f | python ../scripts/logp_score.py > $DIR/results.$i
	python ../scripts/logp_analyze.py --delta 0.6 < $DIR/results.$i
done
