import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import argparse
from collections import deque
import cPickle as pickle

from fast_jtnn import *
import rdkit

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--load_epoch', type=int, default=-1)

parser.add_argument('--hidden_size', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--rand_size', type=int, default=8)
parser.add_argument('--depthT', type=int, default=6)
parser.add_argument('--depthG', type=int, default=3)
parser.add_argument('--share_embedding', action='store_true')
parser.add_argument('--use_molatt', action='store_true')

parser.add_argument('--clip_norm', type=float, default=50.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=1e-3)

args = parser.parse_args()
print args
  
vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

model = DiffVAE(vocab, args).cuda()

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

if args.load_epoch >= 0:
    model.load_state_dict(torch.load(args.save_dir + "/model.iter-" + str(args.load_epoch)))

print "Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
scheduler.step()

PRINT_ITER = 20
param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

for epoch in xrange(args.load_epoch + 1, args.epoch):
    loader = PairTreeFolder(args.train, vocab, args.batch_size, num_workers=4)
    meters = np.zeros(4)

    for it, batch in enumerate(loader):
        x_batch, y_batch = batch
        try:
            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc = model(x_batch, y_batch, args.beta)
            loss.backward()
        except Exception as e:
            print e
            continue

        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()

        meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

        if (it + 1) % PRINT_ITER == 0:
            meters /= PRINT_ITER
            print "KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model))
            sys.stdout.flush()
            meters *= 0

    scheduler.step()

    print "learning rate: %.6f" % scheduler.get_lr()[0]
    if args.save_dir is not None:
        torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(epoch))

