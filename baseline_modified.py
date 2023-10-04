import os.path
import pickle

import nltk
import pandas as pd
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def preprocess_text(text, stop_words):
    """Tokenize, lowercase, and remove stopwords from the text."""
    tokens = word_tokenize(text.lower())
    tok_sent = [word for word in tokens if word not in stop_words and word.isalpha()]
    return tok_sent


def multiprocess_preprocess_joblib(data, column_name, n_jobs):
    """Preprocess data using multiple processes with joblib."""

    # Using joblib's Parallel and delayed to run preprocessing in parallel
    tasks = (delayed(preprocess_text)(text) for text in data[column_name])
    processed_data = Parallel(n_jobs=n_jobs)(tasks)
    return processed_data


def load_train_queries():
    train_queries = None
    if not os.path.isfile('data/preprocessed_queries_train.csv'):
        print('Preprocessing train queries...')
        train_queries = pd.read_csv('data/queries_train.csv', sep=',')
        train_queries['processed_query'] = multiprocess_preprocess_joblib(train_queries, 'query')
        train_queries = train_queries.drop(columns=['query'])
        train_queries = train_queries.rename(columns={'processed_query': 'query'})
        train_queries.to_csv('data/preprocessed_queries_train.csv', index=False)
    else: 
        print('Loading preprocessed train queries...')
        train_queries = pd.read_csv('data/preprocessed_queries_train.csv')
    return train_queries


def load_test_queries():
    test_queries = None
    if not os.path.isfile('data/preprocessed_queries_test.csv'):
        print('Preprocessing test queries...')
        test_queries = pd.read_csv('data/queries_test.csv', sep=',')
        test_queries['processed_query'] = multiprocess_preprocess_joblib(test_queries, 'query')
        test_queries = test_queries.drop(columns=['query'])
        test_queries = test_queries.rename(columns={'processed_query': 'query'})
        test_queries.to_csv('data/preprocessed_queries_test.csv', index=False)
    else:
        print('Loading preprocessed test queries...')
        test_queries = pd.read_csv('data/preprocessed_queries_test.csv')
    return test_queries


def load_dataset():
    dataset = None
    # if data/preprocessed_collection.tsv does not exist, create it
    if not os.path.isfile('data/preprocessed_collection.tsv'):
        print('Preprocessing collection...')
        dataset = pd.read_csv('data/collection.tsv', sep='\t', names=['id', 'text'])
        dataset['processed_text'] = multiprocess_preprocess_joblib(dataset, 'text')
        dataset = dataset.drop(columns=['text'])
        dataset = dataset.rename(columns={'processed_text': 'text'})
        dataset.to_csv('data/preprocessed_collection.tsv', sep='\t', index=False)
        # dropna but preserve ids
        dataset = dataset.dropna()
    else:
        print('Loading preprocessed collection...')
        dataset = pd.read_csv('data/preprocessed_collection.tsv', sep='\t')
        dataset = dataset.dropna()
    return dataset

def load_index(dataset):
    # if pickle index is not present load it
    if not os.path.isfile('data/bm25_index.pkl'):
        print("Index data not present, creating...")
        bm25 = BM25Okapi(dataset['text'])
        with open('data/bm25_index.pickle', 'wb') as handle:
            pickle.dump(bm25, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading index data...")
        with open('data/bm25_index.pkl', 'rb') as f:
            bm25 = pickle.load(f)
    return bm25


def generate_trec_runfile(dataset, ranking_results, run_identifier, output_file):
    """Generate a TREC runfile using the given ranking results."""
    with open(output_file, 'w') as file:
        for qid, passage_indices in ranking_results.items():
            for rank, (passage_idx, score) in enumerate(passage_indices, 1):
                # Construct the turn identifier from the qid
                topic_id, turn_id = qid.split("_")
                turn_identifier = f"{topic_id}_{turn_id}"
                
                # Retrieve the document ID from the collection using the passage index
                doc_id = dataset.iloc[passage_idx]['id']
                
                # Write the formatted line to the file
                file.write(f"{turn_identifier} Q0 {doc_id} {rank} {score} {run_identifier}\n")


def retrieve_rankings_score(row, bm25, top_k):
    qid = row['qid']
    query_text = row['query']

    scores = bm25.get_scores(query_text)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_scores = [scores[i] for i in top_indices]

    return qid, list(zip(top_indices, top_scores))


def main():
    query_type = None
    while query_type not in ['test', 'train']:
        query_type = input("Enter test/train")

    # Downloading stopwords (just a mock step since we don't have internet access here)
    nltk.download('stopwords')
    nltk.download('punkt')

    # We'll use a predefined list of English stopwords.
    stop_words = set(stopwords.words('english'))
    # remove 'about' from stopwords
    stop_words.remove('about')

    # Retrieve top 1000 passages for each query in test_queries
    ranking_results_new = {}
    top_k = 1000
    run_id_new = "BM25_integration_run"

    ranking_results_dict = {}

    dataset = load_dataset()
    bm25 = load_index(dataset)

    queries = load_train_queries() if query_type == 'train' else load_test_queries()
    for _, row in tqdm(queries.iterrows(), total=queries.shape[0]):
        qid, top_indices = retrieve_rankings_score(row, bm25, top_k)
        ranking_results_dict[qid] = top_indices

    # Generate the TREC runfile using the results
    output_filename_parallel = f"data/trec_runfile_{query_type}_parallel.txt"
    generate_trec_runfile(ranking_results_dict, run_id_new, output_filename_parallel)


if __name__ == '__main__':
    main()
