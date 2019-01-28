# Variational Junction Tree Encoder-Decoder

The folder contains the training and testing scripts for variational junction tree encoder-decoder. The trained model is saved in `drd2/`, `qed/`, `logp04/` and `logp06` for respective tasks. Sample translations of the test set are provided in `results/` folder. 

Please make sure that this repo can be found by the system by running
```
export PYTHONPATH=$path_to_repo/iclr19-graph2graph
```

## Preprocessing
For fast training, we need to first preprocess the training data:
```
python ../scripts/preprocess.py --train ../data/logp06/train_pairs.txt --ncpu 8
```
