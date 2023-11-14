from utils import *
import argparse as ap

def main(args):
    if args.model=='b':
        query_type = args.type
        use_rewritten = args.use_rewritten_query

        download_if_not_exists('corpora/stopwords')
        download_if_not_exists('tokenizers/punkt')
        download_if_not_exists('corpora/wordnet')

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
    parser = ap.ArgumentParser()
    parser.add_argument('--preprocess', type=bool, default=False, help='Preprocess the data')
    parser.add_argument('--model', type=str, choices=['a', 'b'], default='a', help='b for baseline and a for advanced model')
    parser.add_argument('--type', type=str, choices=['train', 'test'], default='test', help='Type of queries to convert either "train" or "test"')
    parser.add_argument('--use_rewritten_query', type=bool, default=True, help='Use rewritten queries')
    parser.add_argument('--dataset', type=str, default='../data/', help='Path to the dataset')
    parser.add_argument('--output', type=str, default='../res/', help='Path to the output file')
    args = parser.parse_args()

    main(args)