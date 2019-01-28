# Adversarial Scaffold Regularization

The folder contains the training and testing scripts for variational junction tree encoder-decoder with adversarial training. The trained model is saved in `models/` for all datasets. Sample translations of the test set are provided in `results/` folder. 

## Preprocessing
Please make sure you first preprocess the paired translation data following ([diff_vae/README.md](../diff_vae)). In addtion, we need to preprocess the "good" molecules in the target domain:
```
python ../scripts/preprocess.py --train ../data/qed/target.txt --mode single --ncpu 8
mkdir target-processed
mv tensor* target-processed
```

## Training
Let's use QED task as a running example. Suppose your preprocessed translation pairs are saved in `../data/qed/processed` and target domain molecules are saved in `../data/qed/target-processed`. To train our model, run
```
mkdir -p newmodels/qed
python arae_train.py --train ../data/qed/processed/ --vocab ../data/qed/vocab.txt \ 
--ymols ../data/qed/target-processed/ --save_dir newmodels/qed \ 
--hidden_size 300 --rand_size 8 --epoch 10 --anneal_rate 0.8 | tee newmodels/qed/LOG
```
Here `--ymols` specifies the folder where target domain molecules are stored.

## Validation
After training, you can evaluate different model checkpoints to select the one with the best performance:
```
bash val_scripts/valid_qed.sh models/qed 10
```

## Testing
After finishing cross-validation, you can test the chosen model on the logp04 task by running
```
python decode.py --test ../data/qed/test.txt --vocab ../data/qed/vocab.txt --model models/qed/model.iter-3 | python ../scripts/qed_score > results.qed
python qed_analyze.py < results.qed
```
You can test our models on all four tasks by running
```
bash decode_and_eval.sh
```
or equivalently `bash eval.sh` if the decoded file is given. This should print the model performance as follows:
```
QED task:
Evaluated on 800 samples
success rate 0.615
DRD2 task:
Evaluated on 1000 samples
success rate 0.786
```
Note that the above results are slightly different from the reported value in the paper. In the paper, we run this evaluation with multiple random seeds and report the average performance. Here we run with just one random seed.
