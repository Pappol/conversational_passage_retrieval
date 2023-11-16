# Conversational Passage Retrieval

This is a repo focused on conversational passage retrieval for course DAT640 at the University of Stavanger. 
### Results
All the results are in the [res](res) folder.

# Table of Contents
1. [Conversational Passage Retrieval](#conversational-passage-retrieval)
2. [Downloading Collection](#downloading-collection)
3. [Environment Setup](#environment-setup)
4. [Code Execution](#code-execution)
   - [Running the Baseline](#running-the-baseline)
   - [Re-ranking](#re-ranking)
5. [Main Runner](#main-runner)
   - [Parser Argument Documentation](#parser-argument-documentation)
7. [Question Rewriting Using Chat-GPT4](#question-rewriting-using-chat-gpt4)
8. [SPLADE](#splade)
9. [Evaluation](#evaluation)
10. [Pre Run Files](#pre-run-files)


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
conda env create -f environment.yaml
```

## Code execution

### Running the baseline

```
python3 baseline.py
```


### Re-ranking
For the Re-ranking we used a huggingface BERT model fine-tuned on the MS-MARCO dataset.
In order to run the Re-ranking follow the following steps:
```

python3 reranking.py
```

## Main runner

Alternatively you could run the main file [conversational_passage_retrieval.py](src%conversational_passage_retrieval.py) which will run the baseline given specific parameters.

### Parser Argument Documentation

This document provides detailed information about the command-line arguments available for the parser. These arguments control various aspects of the data processing and model configuration.

### Arguments

### `--pre_process`

- **Type**: `bool`
- **Default**: `False`
- **Description**: Enables or disables preprocessing of the dataset for Splade format.

### `--model`
- **Type**: `str`
- **Choices**: `['a', 'b']`
- **Default**: `'a'`
- **Description**: Selects the model type. Option 'b' refers to the baseline model, and 'a' to the advanced model.

### `--type`
- **Type**: `str`
- **Choices**: `['train', 'test']`
- **Default**: `'train'`
- **Description**: Specifies the type of queries to convert, either 'train' or 'test'.


### `--qr`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Determines whether to use rewritten queries or not.

### `--dataset`
- **Type**: `str`
- **Default**: `'../data/'`
- **Description**: Path to the dataset.


### `--output`
- **Type**: `str`
- **Default**: `'../res/'`
- **Description**: Path for the output file.


### `--ranking`
- **Type**: `str`
- **Choices**: `['bm25', 'splade']`
- **Default**: `'splade'`
- **Description**: Chooses the ranking method to be used.


### `--res`
- **Type**: `str`
- **Default**: `'../res/'`
- **Description**: Path to the preprocessed dataset.

### Question rewriting using Chat-GPT4

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
The file with the rewritten queries is included in the repository [queries_test_qr.csv](data%2Fqueries_test_qr.csv) and [queries_train_qr.csv](data%2Fqueries_train_qr.csv) and can be used directly.

### SPLADE

1. Clone the [SPLADE repository](https://github.com/naver/splade).
2. Create the SPLADE environment according to the instructions.
3. Move our collection to `data/msmarco/full_collection/raw.tsv` in the SPLADE repository.
4. Move our configuration file [config_splade++_cocondenser_ensembledistil_OURS.yaml](config%2Fconfig_splade%2B%2B_cocondenser_ensembledistil_OURS.yaml) to `conf/config_splade++_cocondenser_ensembledistil_OURS.yaml`
5. Prepare the environment:
```
conda activate splade_env
export PYTHONPATH=$PYTHONPATH:$(pwd)
export SPLADE_CONFIG_NAME="config_splade++_cocondenser_ensembledistil_OURS"
```
6. Run indexing:
```
python3 -m splade.index \
  init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil \
  config.pretrained_no_yamlconfig=true \
  config.index_dir=experiments/pre-trained/index
```
7. Move queries in the tsv format to `data/msmarco/dev_queries/raw.tsv` (use [convert_csv_to_tsv.py](src%2Fconvert_csv_to_tsv.py) in case of csv file).
8. Run ranking:
```
python3 -m splade.retrieve \
  init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil \
  config.pretrained_no_yamlconfig=true \
  config.index_dir=experiments/pre-trained/index \
  config.out_dir=experiments/pre-trained/out
```


## Evaluation

For evaluation we are using the TREC eval tool.

```
git clone https://github.com/usnistgov/trec_eval.git
cd trec_eval
make
./trec_eval -c -m recall.1000 -m map -m recip_rank -m ndcg_cut.3 -l2 -M1000 qrels_train.txt {YOUR_TREC_RUNFILE}
```

## Pre run files

In order not to run the whole pipeline, we have included all the results form the different methods in the folder [res](res). Including the scores as a csv file [scores.csv](res%2Fscores.csv) and the jupiter notebook to generate the plots [plots.ipynb](res%2Fplots.ipynb) to visualize the results.
