import pandas as pd
import argparse as ap
import json


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--type', type=str, default='test', help='Type of queries to convert')
    args = parser.parse_args()

    type = args.type

    if type not in ['train', 'test']:
        raise ValueError('Invalid type. Must be either "train" or "test".')

    # Replace these file paths with your actual file paths
    # tsv_input_file_path = f'data/run.queries_{type}_nopretokenized_gpt4_splade.tsv'
    tsv_input_file_path = f'data/{type}_gpt4_splade.json'
    csv_mapping_file_path = f'data/queries_{type}_gpt4.csv'
    output_file_path = f'data/trec_runfile_{type}_qr_splade2.txt'

    # Step 1: Read the CSV file into a DataFrame and create a mapping dictionary
    mapping_df = pd.read_csv(csv_mapping_file_path)
    print(mapping_df.columns)  # Print the columns to confirm their names

    # Create a dictionary mapping the query IDs to the queries
    mapping_dict = dict(zip(mapping_df.index, mapping_df['qid']))
    print(mapping_dict)

    with open(tsv_input_file_path) as f:
        tsv_df = json.load(f)

    print(tsv_df['0'])

    out = []
    #_df = pd.DataFrame(columns=['qid', 'Q0' 'doc_id', 'rank', 'score', 'method'])

    for key in tsv_df.keys():
        query_res = tsv_df[key]
        # order keys by value
        query_res = sorted(query_res.items(), key=lambda x: x[1], reverse=True)
        # add rows to dataframe
        out.append([[mapping_dict[int(key)], 'Q0', query_res[i][0], i, query_res[i][1], 'splade'] for i in range(len(query_res))])

    tsv_df = pd.DataFrame([item for sublist in out for item in sublist], columns=['qid', 'Q0', 'doc_id', 'rank', 'score', 'method'])

    print(tsv_df.head())

    output_df = tsv_df[['qid', 'Q0', 'doc_id', 'rank', 'score', 'method']]
    output_df.to_csv(output_file_path, sep=' ', index=False, header=False, mode='w')
