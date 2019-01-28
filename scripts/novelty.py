import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, required=True)
args = parser.parse_args()

data = [line.split() for line in sys.stdin]
data = [b for a,b,c,d in data]
preds = set(data)

with open(args.target) as f:
    target = [line.strip("\r\n ") for line in f]
target = set(target)

x = len(preds & target)
print len(preds), len(target), x
print 1 - x * 1.0 / len(target)
