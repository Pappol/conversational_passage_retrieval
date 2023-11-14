import pandas as pd


df = pd.read_csv('data/queries_train.csv')
df[['query']].to_csv('data/queries_train_splade.tsv', sep='\t', header=False, index=True)

df = pd.read_csv('data/queries_train_gpt4.csv')
df[['query']].to_csv('data/queries_train_gpt4_splade.tsv', sep='\t', header=False, index=True)
