#!/bin/bash

echo "logp06 task:"
python decode.py --test ../data/logp06/test.txt --vocab ../data/logp06/vocab.txt --model models/logp06/model.iter-5 --rand_size 16 --share_embedding | python ../scripts/logp_score.py > results.logp06
python ../scripts/logp_analyze.py < results.logp06

echo "logp04 task:"
python decode.py --test ../data/logp04/test.txt --vocab ../data/logp04/vocab.txt --model models/logp04/model.iter-5 --hidden_size 330 --share_embedding | python ../scripts/logp_score.py > results.logp04
python ../scripts/logp_analyze.py --delta 0.4 < results.logp04

echo "QED task:"
python decode.py --test ../data/qed/test.txt --vocab ../data/qed/vocab.txt --model models/qed/model.iter-4 --use_molatt | python ../scripts/qed_score.py > results.qed
python ../scripts/qed_analyze.py < results.qed

echo "DRD2 task:"
python decode.py --test ../data/drd2/test.txt --vocab ../data/drd2/vocab.txt --model models/drd2/model.iter-18 --use_molatt | python ../scripts/drd2_score.py > results.drd2
python ../scripts/drd2_analyze.py < results.drd2
