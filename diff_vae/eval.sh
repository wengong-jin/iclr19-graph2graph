#!/bin/bash

echo "logp06 task:"
python ../scripts/logp_analyze.py < results/results.logp06
echo "logp04 task:"
python ../scripts/logp_analyze.py --delta 0.4 < results/results.logp04
echo "QED task:"
python ../scripts/qed_analyze.py < results/results.qed
echo "DRD2 task:"
python ../scripts/drd2_analyze.py < results/results.drd2
