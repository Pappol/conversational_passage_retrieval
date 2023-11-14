import sys
import pandas as pd
import json


def convert_splade_json_to_trec_runfile(json_path, csv_path, out_path):
    # Step 1: Read the CSV file into a DataFrame and create a mapping dictionary
    mapping_df = pd.read_csv(csv_path)
    print(mapping_df.columns)  # Print the columns to confirm their names

    # Create a dictionary mapping the query IDs to the queries
    mapping_dict = dict(zip(mapping_df.index, mapping_df['qid']))
    print(mapping_dict)

    with open(json_path) as f:
        json_df = json.load(f)

    print(json_df['0'])

    out = []
    #_df = pd.DataFrame(columns=['qid', 'Q0' 'doc_id', 'rank', 'score', 'method'])

    for key in json_df.keys():
        query_res = json_df[key]
        # order keys by value
        query_res = sorted(query_res.items(), key=lambda x: x[1], reverse=True)
        # add rows to dataframe
        out.append([[mapping_dict[int(key)], 'Q0', query_res[i][0], i, query_res[i][1], 'splade'] for i in range(len(query_res))])

    json_df = pd.DataFrame([item for sublist in out for item in sublist], columns=['qid', 'Q0', 'doc_id', 'rank', 'score', 'method'])

    print(json_df.head())

    output_df = json_df[['qid', 'Q0', 'doc_id', 'rank', 'score', 'method']]
    output_df.to_csv(out_path, sep=' ', index=False, header=False, mode='w')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Expecting 3 arguments: source json file path, source csv file path and destination TREC runfile path.")

    convert_splade_json_to_trec_runfile(sys.argv[1], sys.argv[2], sys.argv[3])
