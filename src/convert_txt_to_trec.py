import sys
import pandas as pd

def convert_txt_to_trec(src_file, trec_file):
    """Converts csv query input file to tsv. Expecting the 'query' column in the source file."""
    df = pd.read_csv(src_file, sep=' ', header=None)
    df[[0, 2]].to_csv(trec_file, header=['qid', 'docid'], index=False)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Expecting two arguments: source csv file path and destination trec file path.")

    src_file = sys.argv[1]
    dst_file = sys.argv[2]
    convert_txt_to_trec(src_file, dst_file)
