from utils import *


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
