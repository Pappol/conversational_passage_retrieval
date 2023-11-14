import sys
import pandas as pd


def convert_csv_to_tsv(src_file, tsv_file):
    """Converts csv query input file to tsv. Expecting the 'query' column in the source file."""
    df = pd.read_csv(src_file)
    df[['query']].to_csv(tsv_file, sep='\t', header=False, index=True)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Expecting two arguments: source csv file path and destination tsv file path.")

    src_file = sys.argv[1]
    dst_file = sys.argv[2]
    convert_csv_to_tsv(src_file, dst_file)
