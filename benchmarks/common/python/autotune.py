import sys
import pathlib
import argparse
from autotune_find_best_comb import *


def get_lambda_pairs(bench_id_str: str):
    # 0X, 1MM, 2MM, 3CONV, 4CUDA, 5MM, 6MM, 7MM, 8MM
    id = int(bench_id_str)
    if id == 0:
        print(f'The given bench_id of {bench_id_str} is does not need any pairs lambda (get_lambda_pairs).')
        return None
    elif id == 1 or id == 2 or id == 5 or id == 6 or id == 7 or id == 8:
        return lambda pairs: {
            'N': pairs['N'],
            'UNROLL_FACTOR0': pairs['UNROLL_FACTOR0'],
            'UNROLL_FACTOR1': pairs['UNROLL_FACTOR1'],
            'UNROLL_FACTOR2': pairs['UNROLL_FACTOR2']
        }
    elif id == 4:
        print(f'The given bench_id of {bench_id_str} is does not need any pairs lambda (get_lambda_pairs).')
        return None
    else:
        # benchID 3 also falls here
        print(f'The given bench_id of {bench_id_str} is defined yet (get_lambda_pairs).')
        return None


def get_lambda_parse_pairs(bench_id_str: str):
    # 0X, 1MM, 2MM, 3CONV, 4CUDA, 5MM, 6MM, 7MM, 8MM
    id = int(bench_id_str)
    if id == 0:
        print(f'The given bench_id of {bench_id_str} is does not need any pairs lambda (get_lambda_pairs).')
        return None
    elif id == 1 or id == 2 or id == 5 or id == 6 or id == 7 or id == 8:
        return lambda pairs, hw_name: \
            'name=' + pairs['name'] + f';;hw={hw_name}' + ';;N=' + str(pairs['N']) + \
            ';;UNROLL_FACTOR0=' + str(pairs['UNROLL_FACTOR0']) + \
            ';;UNROLL_FACTOR1=' + str(pairs['UNROLL_FACTOR1']) + \
            ';;UNROLL_FACTOR2=' + str(pairs['UNROLL_FACTOR2'])
    elif id == 4:
        print(f'The given bench_id of {bench_id_str} is does not need any pairs lambda (get_lambda_pairs).')
        return None
    else:
        # benchID 3 also falls here
        print(f'The given bench_id of {bench_id_str} is defined yet (get_lambda_pairs).')
        return None


if __name__ == '__main__':
    # Can be used as a standalone script
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps-dir', type=str, required=True)
    parser.add_argument('--benchid', type=str, required=True)
    args = parser.parse_args()
    get_best_config(
        args.dumps_dir,
        args.benchid,
        (pathlib.Path(args.dumps_dir) / "autotuner.json").__str__(),
        get_lambda_pairs(args.benchid),
        get_lambda_parse_pairs(args.benchid)
    )
