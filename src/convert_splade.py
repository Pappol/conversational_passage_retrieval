import pandas as pd
import argparse as ap


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--type', type=str, default='test', help='Type of queries to convert')
    args = parser.parse_args()

    type = args.type

    if type not in ['train', 'test']:
        raise ValueError('Invalid type. Must be either "train" or "test".')

    # Replace these file paths with your actual file paths
    tsv_input_file_path = f'data/run.queries_{type}_custom_collection_gpt4_splade.tsv'
    csv_mapping_file_path = f'data/queries_{type}_gpt4.csv'
    output_file_path = f'data/trec_runfile_{type}_qr_splade.txt'

    # Step 1: Read the CSV file into a DataFrame and create a mapping dictionary
    mapping_df = pd.read_csv(csv_mapping_file_path)
    print(mapping_df.columns)  # Print the columns to confirm their names

    # Create a dictionary mapping the query IDs to the queries
    mapping_dict = dict(zip(mapping_df.index, mapping_df['qid']))
    print(mapping_dict)

    tsv_df = pd.read_csv(tsv_input_file_path, sep='\t', header=None, names=['qid', 'doc_id', 'rank'])
    print(tsv_df.head())

    def map_to_qid(row):
        return mapping_dict.get(row['qid'], 'unknown')

    tsv_df['qid'] = tsv_df.apply(map_to_qid, axis=1)

    print(tsv_df.head())

    tsv_df['Q0'] = 'Q0'
    tsv_df['score'] = 1 + tsv_df['rank'] * -0.001
    tsv_df['method'] = 'SPLADE'

    print(tsv_df.head())

    output_df = tsv_df[['qid', 'Q0', 'doc_id', 'rank', 'score', 'method']]
    output_df.to_csv(output_file_path, sep=' ', index=False, header=False, mode='w')
