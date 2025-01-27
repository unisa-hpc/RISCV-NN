

def get_lambda_pairs(bench_id: int):
    # 0X, 1MM, 2MM, 3CONV, 4CUDA, 5MM, 6MM, 7MM, 8MM
    # make sure bench_id is an integer
    if not isinstance(bench_id, int):
        raise ValueError(f'bench_id should be an integer, got {bench_id}')
    if bench_id == 0:
        print(f'The given bench_id of {bench_id} is does not need any pairs lambda (get_lambda_pairs).')
        return None
    elif bench_id == 1 or bench_id == 2 or bench_id == 5 or bench_id == 6 or bench_id == 7 or bench_id == 8:
        return lambda pairs: {
            'N': pairs['N'],
            'UNROLL_FACTOR0': pairs['UNROLL_FACTOR0'],
            'UNROLL_FACTOR1': pairs['UNROLL_FACTOR1'],
            'UNROLL_FACTOR2': pairs['UNROLL_FACTOR2']
        }
    elif bench_id == 4:
        print(f'The given bench_id of {bench_id} is does not need any pairs lambda (get_lambda_pairs).')
        return None
    else:
        # benchID 3 also falls here
        print(f'The given bench_id of {bench_id} is defined yet (get_lambda_pairs).')
        return None


def get_lambda_parse_pairs(bench_id: int):
    # 0X, 1MM, 2MM, 3CONV, 4CUDA, 5MM, 6MM, 7MM, 8MM
    if not isinstance(bench_id, int):
        raise ValueError(f'bench_id should be an integer, got {bench_id}')
    if bench_id == 0:
        print(f'The given bench_id of {bench_id} is does not need any pairs lambda (get_lambda_pairs).')
        return None
    elif bench_id == 1 or bench_id == 2 or bench_id == 5 or bench_id == 6 or bench_id == 7 or bench_id == 8:
        return lambda pairs, hw_name: \
            'name=' + pairs['name'] + f';;hw={hw_name}' + ';;N=' + str(pairs['N']) + \
            ';;UNROLL_FACTOR0=' + str(pairs['UNROLL_FACTOR0']) + \
            ';;UNROLL_FACTOR1=' + str(pairs['UNROLL_FACTOR1']) + \
            ';;UNROLL_FACTOR2=' + str(pairs['UNROLL_FACTOR2'])
    elif bench_id == 4:
        print(f'The given bench_id of {bench_id} is does not need any pairs lambda (get_lambda_pairs).')
        return None
    else:
        # benchID 3 also falls here
        print(f'The given bench_id of {bench_id} is defined yet (get_lambda_pairs).')
        return None
