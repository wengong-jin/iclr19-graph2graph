#!/bin/bash

DIR=$1
NUM=$2

for ((i=1; i<NUM; i++)); do
	f=$DIR/model.iter-$i
	echo $f
	python decode.py --test ../data/qed/valid.txt --vocab ../data/qed/vocab.txt --model $f | python ../scripts/qed_score.py > $DIR/results.$i
	python ../scripts/qed_analyze.py < $DIR/results.$i
done
