#
# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0
#

def get_tunable_params_list(bench_id: int):
    # 0X, 1MM, 2MM, 3CONV, 4CUDA, 5MM, 6MM, 7MM, 8MM
    if not isinstance(bench_id, int):
        raise ValueError(f'bench_id should be an integer, got {bench_id}')
    if bench_id == 0:
        raise ValueError(f'The given bench_id of {bench_id} is does not need any tunable parameters (get_tunable_params).')
    elif bench_id == 1 or bench_id == 2 or bench_id == 5 or bench_id == 6 or bench_id == 7 or bench_id == 8 or bench_id == 10:
        return ['UNROLL_FACTOR0', 'UNROLL_FACTOR1', 'UNROLL_FACTOR2']
    elif bench_id == 4:
        raise ValueError(f'The given bench_id of {bench_id} is does not need any tunable parameters (get_tunable_params).')
    else:
        # benchID 3 also falls here
        raise ValueError(f'The given bench_id of {bench_id} is not defined yet (get_tunable_params).')

def get_lambda_pairs(bench_id: int):
    # 0X, 1MM, 2MM, 3CONV, 4CUDA, 5MM, 6MM, 7MM, 8MM
    # make sure bench_id is an integer
    if not isinstance(bench_id, int):
        raise ValueError(f'bench_id should be an integer, got {bench_id}')
    if bench_id == 0:
        raise ValueError(f'The given bench_id of {bench_id} is does not need any pairs lambda (get_lambda_pairs).')
    elif bench_id == 1 or bench_id == 2 or bench_id == 5 or bench_id == 6 or bench_id == 7 or bench_id == 8 or bench_id == 10:
        return lambda pairs: {
            'FLAG_AUTOTUNE_DISABLED': pairs['FLAG_AUTOTUNE_DISABLED'],
            'N': pairs['N'],
            'UNROLL_FACTOR0': pairs['UNROLL_FACTOR0'],
            'UNROLL_FACTOR1': pairs['UNROLL_FACTOR1'],
            'UNROLL_FACTOR2': pairs['UNROLL_FACTOR2']
        }
    elif bench_id == 4:
        raise ValueError(f'The given bench_id of {bench_id} is does not need any pairs lambda (get_lambda_pairs).')
    else:
        # benchID 3 also falls here
        raise ValueError(f'The given bench_id of {bench_id} is not defined yet (get_lambda_pairs).')


def get_lambda_parse_pairs(bench_id: int):
    # 0X, 1MM, 2MM, 3CONV, 4CUDA, 5MM, 6MM, 7MM, 8MM
    if not isinstance(bench_id, int):
        raise ValueError(f'bench_id should be an integer, got {bench_id}')
    if bench_id == 0:
        raise ValueError(f'The given bench_id of {bench_id} is does not need any pairs lambda (get_lambda_pairs).')
    elif bench_id == 1 or bench_id == 2 or bench_id == 5 or bench_id == 6 or bench_id == 7 or bench_id == 8 or bench_id == 10:
        return lambda pairs, hw_name: \
            'name=' + pairs['name'] + f';;hw={hw_name}' + ';;N=' + str(pairs['N']) + \
            ';;UNROLL_FACTOR0=' + str(pairs['UNROLL_FACTOR0']) + \
            ';;UNROLL_FACTOR1=' + str(pairs['UNROLL_FACTOR1']) + \
            ';;UNROLL_FACTOR2=' + str(pairs['UNROLL_FACTOR2'])
    elif bench_id == 4:
        raise ValueError(f'The given bench_id of {bench_id} is does not need any pairs lambda (get_lambda_pairs).')
    else:
        # benchID 3 also falls here
        raise ValueError(f'The given bench_id of {bench_id} is not defined yet (get_lambda_pairs).')
