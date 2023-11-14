import pandas as pd


df_orig = pd.read_csv('../data/queries_train.csv')
df_output = pd.read_csv('../data/run.queries_train_splade.tsv', sep='\t')

# TODO: merge it together
