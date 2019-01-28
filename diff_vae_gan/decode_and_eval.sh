#!/bin/bash

echo "QED task:"
python decode.py --test ../data/qed/test.txt --vocab ../data/qed/vocab.txt --model models/qed/model.iter-3 --use_molatt | python ../scripts/qed_score.py > results.qed
python ../scripts/qed_analyze.py < results.qed

echo "DRD2 task:"
python decode.py --test ../data/drd2/test.txt --vocab ../data/drd2/vocab.txt --model models/drd2/model.iter-20 --use_molatt | python ../scripts/drd2_score.py > results.drd2
python ../scripts/drd2_analyze.py < results.drd2
