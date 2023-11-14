import numpy as np
import os.path
import pickle
import nltk
import pandas as pd
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import re


def preprocess_text(text, stop_words, stemmer):
    """
    This function preprocesses the text by keeping only alphabetic characters,
    removing stopwords and lowercasing the text.
    """

    # lowercase and remove non-alphabetic characters (even numbers) 
    clean_text = re.sub(r'[^a-z]', ' ', text.lower())

    # tokenize and remove stopwords
    tokens = word_tokenize(clean_text)
    tok_sent = [word for word in tokens if word not in stop_words and word.isalpha()]

    # apply the lemmatization
    tok_sent = [stemmer.lemmatize(word) for word in tok_sent]



    if len(tok_sent) == 0:
        # print("Empty tokienized string: ", text)
        # if there is no token left after preprocessing, return 'and' as a token
        # this is because we know that the word 'and' is removed from the stopwords
        # and we use it as a special token
        return ['and'] 
    return tok_sent #' '.join(tok_sent)


def multiprocess_preprocess_joblib(data, column_name, stop_words, stemmer, n_jobs=30):
    """Preprocess data using multiple processes to speed up the work"""

    # using parallel and delayed to run preprocessing in parallel
    tasks = (delayed(preprocess_text)(text, stop_words, stemmer) for text in data[column_name])
    processed_data = Parallel(n_jobs=n_jobs)(tasks)
    return processed_data


def load_train_queries(stop_words, stemmer, use_rewritten_query=False):
    """
    This function loads the train queries and preprocesses them.
    """
    filename = 'queries_train_gpt4' if use_rewritten_query else 'queries_train'
    # look if preprocessed queries are present
    if not os.path.isfile(f'data/preprocessed_{filename}.tsv'):
        print('Preprocessing train queries...')
        train_querys = pd.read_csv(f'data/{filename}.csv', sep=',')
        train_querys['processed_query'] = multiprocess_preprocess_joblib(train_querys, 'query', stop_words, stemmer)
        # rename columns
        train_querys = train_querys.drop(columns=['query'])
        train_querys = train_querys.rename(columns={'processed_query': 'query'})
        # save preprocessed collection to tsv file (for correct format of lists)
        train_querys.to_csv(f'data/preprocessed_{filename}.tsv', sep='\t', index=False)
    else:
        # if preprocessed queries are present, load them
        print('Loading preprocessed train queries...')
        train_querys = pd.read_csv(f'data/preprocessed_{filename}.tsv', sep='\t')
    return train_querys

def load_test_queries(stop_words, stemmer, use_rewritten_query):
    """
    This function loads the test queries and preprocesses them.
    """
    filename = 'queries_test_gpt4' if use_rewritten_query else 'queries_test'

    # look if preprocessed queries are present
    if not os.path.isfile(f'data/preprocessed_{filename}.tsv'):
        print('Preprocessing test queries...')
        test_querys = pd.read_csv(f'data/{filename}.csv', sep=',')
        test_querys['processed_query'] = multiprocess_preprocess_joblib(test_querys, 'query', stop_words, stemmer)
        # rename columns
        test_querys = test_querys.drop(columns=['query'])
        test_querys = test_querys.rename(columns={'processed_query': 'query'}) 
        # save preprocessed collection to tsv file (for correct format of lists)
        test_querys.to_csv(f'data/preprocessed_{filename}.tsv', sep='\t', index=False)
    else:
        # if preprocessed queries are present, load them
        print('Loading preprocessed test queries...')
        test_querys = pd.read_csv(f'data/preprocessed_{filename}.tsv', sep='\t')
    return test_querys

def load_dataset(stop_words, stemmer):
    """
    This function loads the collection and preprocesses it.
    """

    # look if preprocessed collection is present
    if not os.path.isfile('data/preprocessed_collection.tsv'):
        print('Preprocessing collection...')
        dataset = pd.read_csv('data/collection.tsv', sep='\t', names=['id', 'text'])
        dataset['processed_text'] = multiprocess_preprocess_joblib(dataset, 'text', stop_words, stemmer)
        # rename columns
        dataset = dataset.drop(columns=['text'])
        dataset = dataset.rename(columns={'processed_text': 'text'})
        # save preprocessed collection to tsv file (for correct format of lists)
        dataset.to_csv('data/preprocessed_collection.tsv', sep='\t', index=False)
    else:
        # if preprocessed collection is present, load it
        print('Loading preprocessed collection...')
        dataset = pd.read_csv('data/preprocessed_collection.tsv', sep='\t')
    return dataset


def load_index(dataset):
    """
    This function loads the index or creates it if it is not present.
    """
    # if pickle index is not present load it
    if not os.path.isfile('data/bm25_index.pkl'):
        print("Index data not present, creating...")
        # create the index with the BM25Okapi class
        bm25 = BM25Okapi(dataset['text'])
        with open('data/bm25_index.pkl', 'wb') as handle:
            pickle.dump(bm25, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading index data...")
        with open('data/bm25_index.pkl', 'rb') as f:
            bm25 = pickle.load(f)
    return bm25


def generate_trec_runfile(dataset, ranking_results, run_identifier, output_file):
    """
    Generate a TREC runfile using the given ranking results.
    """
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
    """
    This function retrieves the top k passages for a given query.
    """
    qid = row['qid']
    query_text = row['query']

    scores = bm25.get_scores(query_text)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_scores = [scores[i] for i in top_indices]

    return qid, list(zip(top_indices, top_scores))


def main():
    query_type = None
    while query_type not in ['test', 'train']:
        query_type = input("Enter test/train: ")

    use_rewritten = None
    while use_rewritten not in ['true', 'false']:
        use_rewritten = input("Enter true or false for query rewriting: ")
    
    use_rewritten = True if use_rewritten == 'true' else False

    # Downloading stopwords (just a mock step since we don't have internet access here)
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    # We'll use a predefined list of English stopwords.
    stop_words = set(stopwords.words('english'))
    stemmer = nltk.stem.WordNetLemmatizer()

    # Retrieve top 1000 passages for each query in test_queries
    top_k = 1000
    run_id_new = "BM25_integration_run"

    ranking_results_dict = {}

    dataset = load_dataset(stop_words, stemmer)
    bm25 = load_index(dataset)

    queries = load_train_queries(stop_words, stemmer, use_rewritten) if query_type == 'train' else load_test_queries(stop_words, stemmer, use_rewritten)

    # loop over the queries and retrieve the top 1000 passages for each query
    for _, row in tqdm(queries.iterrows(), total=queries.shape[0]):
        qid, top_indices = retrieve_rankings_score(row, bm25, top_k)
        ranking_results_dict[qid] = top_indices

    # Generate the TREC runfile using the results
    qr_text = '_qr' if use_rewritten else ''
    output_filename_parallel = f"data/trec_runfile_{query_type}{qr_text}_parallel.txt"
    generate_trec_runfile(dataset, ranking_results_dict, run_id_new, output_filename_parallel)


if __name__ == '__main__':
    main()
