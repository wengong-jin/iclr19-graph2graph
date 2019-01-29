# Variational Junction Tree Encoder-Decoder

The folder contains the training and testing scripts for variational junction tree encoder-decoder. The trained model is saved in `models/` for all datasets. Sample translations of the test set are provided in `results/` folder. 

Please make sure that this repo can be found by the system by running
```
export PYTHONPATH=$path_to_repo/iclr19-graph2graph
```

## Preprocessing
For fast training, we need to first preprocess the training data. For instance, you can preprocess the training set of logp04 task using:
```
python ../scripts/preprocess.py --train ../data/logp04/train_pairs.txt --ncpu 8
mkdir processed
mv tensor* processed
```
This will put all binary files into `processed/` folder. To preprocess all the datasets, run
```
bash preprocess.sh
```
This will create four folders in `../data/*/processed` for four tasks respectively.

## Vocabulary Extraction (Optional)
The fragment vocabulary files have already been provided in `data/` for all datasets. If you are working on your own dataset, you will need to extract the corresponding fragment vocabulary. Supposed your molecule file is `mols.txt`, which has a SMILES string in each line, then you can run the following script for vocabulary extraction:
```
python ../fast_jtnn/mol_tree.py < mols.txt
```

## Training
Suppose the preprocessed data of logp04 is saved in `processed/` folder. You can train our model on the logp04 task by
```
mkdir -p newmodels/logp04
python vae_train.py --train processed/ --vocab ../data/logp04/vocab.txt --save_dir newmodels/logp04 \
--hidden_size 330 --rand_size 8 --epoch 10 --anneal_rate 0.8 | tee newmodels/logp04/LOG
```
Specifically, `--hidden_size 330` sets the hidden state dimension to be 330 and `--rand_size 8` sets the latent code dimension to be 8. `--epoch 10 --anneal_rate 0.8` means the model will be trained for 10 epochs, with learning rate annealing 0.8. The model checkpoints are saved in `newmodels/logp04`.

The hyperparameters of our experiments in the paper are provided in `train.sh`. Our models with the best validation performance are provided in `models/` folder (e.g., `models/logp04/model.iter-5` for the logp04 model). After you have preprocessed all four datasets, you can run `bash train.sh` to train our model on all datasets:
```
bash train.sh
```

## Validation
After training, you can evaluate different model checkpoints to select the one with the best performance:
```
bash val_scripts/valid_logp04.sh models/logp04 10
```

## Testing
After finishing cross-validation, you can test the chosen model on the logp04 task by running
```
python decode.py --test ../data/logp04/test.txt --vocab ../data/logp04/vocab.txt --model models/logp04/model.iter-5 | python ../scripts/logp_score > results.logp04
python logp_analyze.py --delta 0.4 < results.logp04
```
You can test our models on all four tasks by running
```
bash decode_and_eval.sh
```
or equivalently `bash eval.sh` if the decoded file is given. This should print the model performance as follows:
```
logp06 task: Evaluated on 800 samples
average improvement 2.33211345528 stdev 1.28808721418
logp04 task: Evaluated on 800 samples
average improvement 3.58104201902 stdev 1.62546251435
QED task: Evaluated on 800 samples
success rate 0.605
DRD2 task: Evaluated on 1000 samples
success rate 0.781
```
Note that the above results are slightly different from the reported value in the paper. In the paper, we run this evaluation with multiple random seeds and report the average performance. Here we run with just one random seed.
