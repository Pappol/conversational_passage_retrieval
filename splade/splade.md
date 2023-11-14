Source: https://github.com/castorini/pyserini/blob/master/docs/experiments-spladev2.md

# TRAIN QUERIES
```
python -m pyserini.search.lucene \
  --index indexes/lucene-index.msmarco-passage-distill-splade-max \
  --topics ../conversational_passage_retrieval/data/queries_train_splade.tsv \
  --encoder distill-splade-max \
  --output runs/run.queries_train_splade.tsv \
  --output-format msmarco \
  --batch 36 --threads 12 \
  --hits 1000 \
  --impact
```

# TRAIN QUERIES GPT4
```
python -m pyserini.search.lucene \
  --index indexes/lucene-index.msmarco-passage-distill-splade-max \
  --topics ../conversational_passage_retrieval/data/queries_train_gpt4_splade.tsv \
  --encoder distill-splade-max \
  --output runs/run.queries_train_gpt4_splade.tsv \
  --output-format msmarco \
  --batch 36 --threads 12 \
  --hits 1000 \
  --impact
```

# TEST QUERIES
```
python -m pyserini.search.lucene \
  --index indexes/lucene-index.msmarco-passage-distill-splade-max \
  --topics ../conversational_passage_retrieval/data/queries_test_splade.tsv \
  --encoder distill-splade-max \
  --output runs/run.queries_test_splade.tsv \
  --output-format msmarco \
  --batch 36 --threads 12 \
  --hits 1000 \
  --impact
```

# TEST QUERIES GPT4
```
python -m pyserini.search.lucene \
  --index indexes/lucene-index.msmarco-passage-distill-splade-max \
  --topics ../conversational_passage_retrieval/data/queries_test_gpt4_splade.tsv \
  --encoder distill-splade-max \
  --output runs/run.queries_test_gpt4_splade.tsv \
  --output-format msmarco \
  --batch 36 --threads 12 \
  --hits 1000 \
  --impact
```






python -m pyserini.search.lucene \
  --index indexes/lucene-index.msmarco-custom-collection-distill-splade-max \
  --topics ../conversational_passage_retrieval/data/queries_train_gpt4_splade.tsv \
  --encoder distill-splade-max \
  --output runs/run.queries_train_custom_collection_gpt4_splade.tsv \
  --output-format msmarco \
  --batch 36 --threads 12 \
  --hits 1000 \
  --impact