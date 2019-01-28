#!/bin/bash

echo "QED task:"
python ../scripts/qed_analyze.py < results/results.qed
echo "DRD2 task:"
python ../scripts/drd2_analyze.py < results/results.drd2
