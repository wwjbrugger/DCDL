import pickle as pk
import sys
import pandas as pd
from tabulate import tabulate

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
with open(sys.argv[1], 'rb') as f:
    t=pk.load(f)#.astype(float)

print(tabulate(t, headers='keys', tablefmt='psql'))

#print(tabulate(t.round(2), headers='keys', tablefmt='psql'))

#print(tabulate(t.mean(axis=0).to_frame(name="mean"), headers='keys', tablefmt='psql'))

