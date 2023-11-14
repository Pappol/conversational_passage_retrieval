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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict

def preprocess_text(text, stop_words, stemmer):
    """
    This function preprocesses the text by keeping only alphabetic characters,
    removing stopwords and lowercasing the text.

    Parameters:
    - text (str): The input text to be preprocessed.
    - stop_words (set): A set of words to be excluded from the text.
    - stemmer (Stemmer): An instance of a stemmer/lemmatizer to apply to the words.

    Returns:
    - list: A list of preprocessed words from the input text. If no words remain after preprocessing, 
        the function returns ['and'] as a default value.
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
    """
    Preprocess data using multiple processes to speed up the work

    Parameters:
    - data (pandas.DataFrame): The dataset containing the text to be preprocessed.
    - column_name (str): The name of the column in 'data' that contains the textual data.
    - stop_words (set): A set of stopwords to be removed during preprocessing.
    - stemmer (Stemmer/Lemmatizer object): An object with a 'lemmatize' method for lemmatizing the words in the text.
    - n_jobs (int, optional): The number of jobs to run in parallel. Default is 30.

    Returns:
    - list: A list where each element corresponds to the preprocessed text of each row in the specified column of 'data'.
    
    """

    # using parallel and delayed to run preprocessing in parallel
    tasks = (delayed(preprocess_text)(text, stop_words, stemmer) for text in data[column_name])
    processed_data = Parallel(n_jobs=n_jobs)(tasks)
    return processed_data


def load_train_queries(stop_words, stemmer, use_rewritten_query=False):
    """
    This function loads the train queries and preprocesses them.

    Parameters:
    - stop_words (set): A set of stopwords to be removed during preprocessing.
    - stemmer (Stemmer/Lemmatizer object): An object with a 'lemmatize' method for lemmatizing the words in the text.
    - use_rewritten_query (bool, optional): A flag to determine which file to load the queries from. 
      If True, the function loads from 'queries_train_gpt4.csv', otherwise from 'queries_train.csv'. Default is False.

    Returns:
    - pandas.DataFrame: A DataFrame containing the preprocessed queries. If the preprocessed file is present, it loads 
      from there, otherwise, it preprocesses and then returns the data.

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

    Parameters:
    - stop_words (set): A set of stopwords to be removed during preprocessing.
    - stemmer (Stemmer/Lemmatizer object): An object with a 'lemmatize' method for lemmatizing the words in the text.
    - use_rewritten_query (bool): A flag to determine which file to load the queries from. 
      If True, the function loads from 'queries_test_gpt4.csv', otherwise from 'queries_test.csv'.

    Returns:
    - pandas.DataFrame: A DataFrame containing the preprocessed queries. If the preprocessed file is present, 
      it loads from there, otherwise, it preprocesses and then returns the data.

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

    Parameters:
    stop_words (list of str): A list of words to be excluded during preprocessing.
    stemmer (Stemmer object): An object that provides stemming functionality to reduce words to their root form.

    Returns:
    pandas.DataFrame: A dataframe containing the preprocessed text data. If a preprocessed collection already exists,
                      it is loaded directly; otherwise, the collection is preprocessed and then returned.
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

    Parameters:
    dataset (pandas.DataFrame): A dataframe containing the text data used to create the BM25 index.
                                The dataframe must contain a 'text' column.

    Returns:
    BM25Okapi: An object representing the BM25 index. If an index already exists in the specified path,
               it is loaded; otherwise, a new index is created from the provided dataset and then returned.
    
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
    Parameters:
    dataset (pandas.DataFrame): A dataframe containing the document collection, with each document's unique identifier in the 'id' column.
    ranking_results (dict): A dictionary where each key is a query identifier and its value is a list of tuples. 
                            Each tuple contains a passage index (int) and a corresponding relevance score (float).
    run_identifier (str): A unique identifier for the run, used in the runfile to differentiate results from different systems or configurations.
    output_file (str): The path and filename where the runfile will be saved.

    Returns:
    None: The function does not return anything. It writes the TREC-formatted results to the specified output file.
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

    Parameters:
    row (pandas.Series): A row from a dataframe containing at least two columns: 'qid' for the query identifier and 'query' for the query text.
    bm25 (BM25Okapi): A BM25Okapi object used to score and rank documents based on the given query.
    top_k (int): The number of top-ranked passages to retrieve.

    Returns:
    tuple: A tuple containing two elements:
           1. qid (int or str): The query identifier from the input row.
           2. list of tuples: Each tuple in the list contains two elements:
              a. passage index (int): The index of the passage in the ranking.
              b. score (float): The BM25 score of the passage for the given query.
           The list contains the top k passages ranked according to their BM25 scores.
    """

    qid = row['qid']
    query_text = row['query']

    scores = bm25.get_scores(query_text)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_scores = [scores[i] for i in top_indices]

    return qid, list(zip(top_indices, top_scores))

def download_if_not_exists(package):
    """
    Checks if a specified NLTK package is already downloaded and available locally. 
    If the package is not available, it downloads the package.

    Parameters:
    package (str): The NLTK package name to check and download if necessary. 
                   The package name should be specified in the format 'category/package_name', 
                   for example, 'corpora/stopwords' for the stopwords package, 
                   'tokenizers/punkt' for the Punkt Tokenizer Models, and 
                   'corpora/wordnet' for the WordNet.

    Returns:
    None: The function does not return any value. It prints a message indicating whether 
          the package was already downloaded or has been downloaded during the execution.
    """
    
    try:
        # Check if the package is already downloaded
        nltk.data.find(package)
        print(f"'{package}' is already downloaded.")
    except LookupError:
        # If not, download the package
        nltk.download(package.split('/')[1])
        print(f"Downloaded '{package}'.")