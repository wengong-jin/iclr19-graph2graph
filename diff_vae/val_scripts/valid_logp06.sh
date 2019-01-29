#!/bin/bash

DIR=$1
NUM=$2

for ((i=1; i<NUM; i++)); do
	f=$DIR/model.iter-$i
	if [ -e $f ]; then
		echo $f
		python decode.py --test ../data/logp06/valid.txt --vocab ../data/logp06/vocab.txt --model $f --rand_size 16 --share_embedding | python ../scripts/logp_score.py > $DIR/results.$i
		python ../scripts/logp_analyze.py --delta 0.6 < $DIR/results.$i
	fi
done
