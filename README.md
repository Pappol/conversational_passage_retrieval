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

Rewritten queries are already stored among the other input data.
[queries_test_qr.csv](data%2Fqueries_test_qr.csv)
[queries_train_qr.csv](data%2Fqueries_train_qr.csv)


To run the question rewriting using Chat-GPT4, you can use our custom GPT instance at the following link: https://chat.openai.com/g/g-R920EZnY6-query-rewriting-gpt

If you do not have the Plus subscription you can still give a try with the free ChatGPT3.5 using the following message as first message:
``` 
You are tasked with the role of " Query Rewriter " for a
Conversational Passage Retrieval system . In conversational
queries , subsequent questions may lack essential details
present in prior interactions . Your goal is to integrate
context from previous queries to rewrite the current query
into a more detailed and standalone search query . This will
ensure that the rewritten query is optimized for retrieving
the most relevant passage , even without the conversational
context .
Example :
Conversational Queries :
- Tell me about the International Linguistics Olympiad .
- How do I prepare for it ?
- How tough is the exam ?
Rewritten Queries :
- International Linguistics Olympiad overview .
- Preparation methods for the International Linguistics Olympiad .
- Difficulty level of the International Linguistics Olympiad exam .
```

In both cases you will have to copy the content of every set of query for each topic in the chat window and copy the output in a new file. GPT-4 can handle all the queries at once, but we did not tried with GPT-3.5.

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
