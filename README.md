# Conversational Passage Retrieval

This is a repo focused on conversational passage retrieval for course DAT640 at the University of Stavanger. 

## Downloading collection
```
mkdir data
wget --output-document data/msmarco-passage.tar.gz https://gustav1.ux.uis.no/dat640/msmarco-passage.tar.gz
tar -xzvf data/msmarco-passage.tar.gz -C data/
rm msmarco-passage.tar.gz
```

## Environment setup

To install from environment file run:
```
conda env create -f environment.yml
```

## Running the baseline

```
python3 baseline.py
```

## Question rewriting using Chat-GPT4

Output is stored among the other input data.
[queries_test_qr.csv](data%2Fqueries_test_qr.csv)
[queries_train_qr.csv](data%2Fqueries_train_qr.csv)

## SPLADE



Move our collection to `data/msmarco/full_collection/raw.tsv`


```
conf/config_splade++_cocondenser_ensembledistil_OURS.yaml
```

## Evaluation

For evaluation we are using the TREC eval tool.

```
git clone https://github.com/usnistgov/trec_eval.git
cd trec_eval
make
./trec_eval -c -m recall.1000 -m map -m recip_rank -m ndcg_cut.3 -l2 -M1000 qrels_train.txt {YOUR_TREC_RUNFILE}
```
