from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict
import torch

def get_relevance_dict(path):
    """
    This function returns a dictionary with key = query_id and value = list of relevant doc_id 
    """

    results = pd.read_csv(path, sep=' ', header=None)
    results.columns = ['query_id', 'Q0', 'doc_id', 'rank', 'score', 'run_name']

    # create a dictionary with key = query_id and value = list of doc_id
    # the list of doc_id is ordered by rank

    res_dict = defaultdict(list)    
    for _, row in results.iterrows():
        res_dict[row['query_id']].append(row['doc_id'])

    return res_dict


def rerank(out_path, res_dict, queries, collection, tokenizer, model):
    """
    This function reranks the documents in the runfile using the given model.
    """

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)

    for key, doc_ids in tqdm(res_dict.items()):
        # if the file already contains the key, skip it
        with open(out_path, 'r') as f:
            if key in f.read():
                print(f'Query {key} already in the file')
                continue
            
        # read the query
        query = queries.loc[key]['query']
        scores = {}
        for doc_id in doc_ids:
            # read the document
            doc = collection.loc[doc_id]['text']
            # encode the query and the document
            encoding = tokenizer(query, doc, return_tensors='pt', truncation=True).to(device)

            # truncate the document to 512 tokens
            if encoding['input_ids'].shape[1] > 512:
                encoding['input_ids'] = encoding['input_ids'][:, :512]
                encoding['token_type_ids'] = encoding['token_type_ids'][:, :512]
                encoding['attention_mask'] = encoding['attention_mask'][:, :512]

            # rerank the document
            output = model(**encoding)
            # update the score in the dataframe
            scores[doc_id] = output.logits[0][1].item()

        # sort the dictionary by value
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # update the file with the ordered documents
        with open(out_path, 'a') as f:
            for i, (doc_id, score) in enumerate(sorted_scores):
                f.write(f'{key} Q0 {doc_id} {i+1} {score} bert \n')


def main():
    type = None
    while type not in ['test', 'train']:
        type = input("Enter test/train: ")

    use_rewritten = None
    while use_rewritten not in ['true', 'false']:
        use_rewritten = input("Enter true or false for query rewriting: ")

    use_rewritten = True if use_rewritten == 'true' else False

    ranker = None
    while ranker not in ['bm25', 'splade']:
        ranker = input("Enter bm25 or splade: ")
        ranker = '_' + ranker

    qr_text = '_qr' if use_rewritten == 'true' else ''

    # path for the runfile
    res_path = f'res/runfile_{type}{qr_text}{ranker}.txt'
    # path for the query
    query_path = f'data/queries_{type}{qr_text}.csv'
    # path for the documents
    docs_path = 'data/collection.tsv'
    # path where to save the file
    out_path = f'data/runfile_{type}{qr_text}{ranker}_rr.txt'

    # load the model from huggingface
    tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
    model = AutoModelForSequenceClassification.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")

    # load the queries
    queries = pd.read_csv(query_path)
    queries = queries.set_index('qid')

    # load the collection
    collection = pd.read_csv(docs_path, sep='\t', header=None)
    collection.columns = ['doc_id', 'text']
    collection = collection.set_index('doc_id')

    # load the runfile
    res_dict = get_relevance_dict(res_path)

    # rerank the documents
    rerank(out_path, res_dict, queries, collection, tokenizer, model)


if __name__ == '__main__':
    main()