import pstats
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('input_file')
    args = parser.parse_args()

    sts = pstats.Stats(args.input_file)
    sts.strip_dirs().sort_stats('tottime').print_stats()

