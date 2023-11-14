from utils import *
import argparse as ap

def main(args):
    continue

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--preprocess', type=bool, default=False, help='Preprocess the data')
    parser.add_argument('--model', type=str, default='a', help='b for baseline and a for advanced model')
    args = parser.parse_args()

    main(args)