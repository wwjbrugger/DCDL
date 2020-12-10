import pickle as pk
import sys
import json

path_to_read = sys.argv[1]

with open(path_to_read , 'rb') as f:
    x=pk.load(f)
print(x)