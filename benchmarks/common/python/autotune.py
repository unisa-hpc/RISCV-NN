import sys
import pathlib
import argparse
from autotune_find_best_comb import *
from parsing.lamda_funcs import *


if __name__ == '__main__':
    # Can be used as a standalone script
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps-dir', type=str, required=True)
    parser.add_argument('--benchid', type=str, required=True)
    args = parser.parse_args()
    get_best_config(
        args.dumps_dir,
        int(args.benchid),
        (pathlib.Path(args.dumps_dir) / "autotuner.json").__str__(),
        get_lambda_pairs(int(args.benchid)),
        get_lambda_parse_pairs(int(args.benchid))
    )
