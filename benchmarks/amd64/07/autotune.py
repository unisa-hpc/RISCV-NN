import sys
import pathlib
import argparse
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2].joinpath('common')))
from autotune_find_best_comb import *


if __name__ == '__main__':
    # Can be used as a standalone script
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps-dir', type=str, required=True)
    parser.add_argument('--benchid', type=str, required=True)
    args = parser.parse_args()
    get_best_config(
        args.dumps_dir,
        args.benchid,
        (pathlib.Path(args.dumps_dir)/"autotuner.json").__str__(),
        lambda pairs: {
           'N': pairs['N'],
           'UNROLL_FACTOR0': pairs['UNROLL_FACTOR0'],
           'UNROLL_FACTOR1': pairs['UNROLL_FACTOR1'],
           'UNROLL_FACTOR2': pairs['UNROLL_FACTOR2']
        },
        lambda pairs, hw_name: \
           'name=' + pairs['name'] + f';;hw={hw_name}' + ';;N=' + str(pairs['N']) +
           ';;UNROLL_FACTOR0=' + str(pairs['UNROLL_FACTOR0']) +
           ';;UNROLL_FACTOR1=' + str(pairs['UNROLL_FACTOR1']) +
           ';;UNROLL_FACTOR2=' + str(pairs['UNROLL_FACTOR2'])
    )

