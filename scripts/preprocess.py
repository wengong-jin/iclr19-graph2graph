import torch
import torch.nn as nn
from multiprocessing import Pool

import math, random, sys
import cPickle as pickle
import argparse

from fast_jtnn import *
import rdkit

def tensorize(smiles, assm=False):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol
        del node.clique

    return mol_tree

def tensorize_pair(smiles_pair):
    mol_tree0 = tensorize(smiles_pair[0], assm=False)
    mol_tree1 = tensorize(smiles_pair[1], assm=True)
    return (mol_tree0, mol_tree1)

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--mode', type=str, default='pair')
    parser.add_argument('--ncpu', type=int, default=8)
    args = parser.parse_args()

    pool = Pool(args.ncpu)

    if args.mode == 'pair':
        #dataset contains molecule pairs
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[:2] for line in f]

        all_data = pool.map(tensorize_pair, data)
        num_splits = len(data) / 10000

        le = (len(all_data) + num_splits - 1) / num_splits

        for split_id in xrange(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open('tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

    elif args.mode == 'single':
        #dataset contains single molecules
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[0] for line in f]

        all_data = pool.map(tensorize, data)
        num_splits = len(data) / 10000

        le = (len(all_data) + num_splits - 1) / num_splits

        for split_id in xrange(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open('tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

