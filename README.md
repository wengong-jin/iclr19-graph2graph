# Learning Multimodal Graph-to-Graph Translation for Molecular Optimization

This is the official implementation of junction tree encoder-decoder model in https://arxiv.org/abs/1812.01070

## Requirements
* Python == 2.7
* RDKit >= 2017.09
* PyTorch >= 0.4.0
* Numpy
* scikit-learn

The code has been tested under python 2.7 with pytorch 0.4.1. 

## Quick Start
A quick summary of different folders:
* `data/` contains the training, validation and test set of logP, QED and DRD2 tasks described in the paper.
* `fast_jtnn/` contains the implementation of junction tree encoder-decoder.
* `diff_vae/` includes the training and decoding script of variational junction tree encoder-decoder ([README](./diff_vae)).
* `diff_vae_gan/` includes the training and decoding script of adversarial training module ([README](./diff_vae_gan)).
* `props/` is the property evaluation module, including penalized logP, QED and DRD2 property calculation.
* `scripts/` provides evaluation and data preprocessing scripts.

## Contact
Wengong Jin (wengong@csail.mit.edu)
