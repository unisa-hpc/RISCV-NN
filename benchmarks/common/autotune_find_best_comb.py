from matplotlib import pyplot as plt
import argparse
import json
import pathlib
import pandas as pd
import seaborn as sns
import sys
from matplotlib.lines import Line2D

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2].joinpath('common')))
from timer_stats import TimerStatsParser


def get_all_hw_names(dump_dir: str, bench_id: str) -> [str]:
    """
    Get the abs path of all the json files in the dump directory, recursively.
    """
    # read all the lines from the file at dump_dir/benchId.txt
    with open(pathlib.Path(dump_dir).joinpath(f'benchId{bench_id}.txt'), 'r') as f:
        lines = f.readlines()

    hw_names = []
    for line in lines:
        # skip if line is empty
        if not line.strip():
            continue
        # find all the json files in current_run_dir
        line = line.strip()
        # seperate hw, path for each line
        hw, line = line.split(',', 1)
        if hw not in hw_names:
            hw_names.append(hw)

    print(f'Found these hardware names: {hw_names}')
    return hw_names


def get_all_json_files(dump_dir: str, bench_id: str, only_this_hw: str) -> [str]:
    """
    Get the abs path of all the json files in the dump directory, recursively.
    """
    # read all the lines from the file at dump_dir/benchId.txt
    with open(pathlib.Path(dump_dir).joinpath(f'benchId{bench_id}.txt'), 'r') as f:
        lines = f.readlines()

    json_files = []
    for line in lines:
        # skip if line is empty
        if not line.strip():
            continue

        # find all the json files in line
        line = line.strip()

        # seperate hw, path for each line
        hw, sub_dump_dir_name = line.split(',', 1)
        sub_dump_dir_name = sub_dump_dir_name.strip()

        if only_this_hw != hw:
            continue

        sub_dump_dir_path = pathlib.Path(dump_dir) / pathlib.Path(sub_dump_dir_name)
        json_files.extend([str(f) for f in sub_dump_dir_path.rglob('*.json')])

    print(f'Found {len(lines)} sub-dump directories and a '
          f'total of {len(json_files)} json files for benchmark ID {bench_id}')
    return json_files


def get_best_config(dumps_dir: str, benchid: str):
    """
    Find the best unroll factor for each hardware and print it.
    It looks for benchId{benchid}.txt in the dumps_dir to extract all the relevant json files.
    """
    all_hw_names = get_all_hw_names(dumps_dir, benchid)
    for hw_name in all_hw_names:
        print('=================================================')
        print('Finding best unroll factor for hardware:', hw_name)

        all_jsons = get_all_json_files(dumps_dir, benchid, hw_name)

        parse_pairs = lambda pairs: {
            'N': pairs['N'],
            'unroll_factor': pairs['unroll_factor']
        }
        unique_names = lambda pairs: \
            'name=' + pairs['name'] + f';;hw={hw_name}' + ';;N=' + str(pairs['N']) + ';;unroll=' + str(
                pairs['unroll_factor'])
        parse_unique_names = lambda name: {
            pair.split('=')[0]: pair.split('=')[1] for pair in name.split(';;')
        }

        parsed_runs = [TimerStatsParser(j, parse_pairs) for j in all_jsons]
        parsed_union = pd.concat([run.get_df() for run in parsed_runs], ignore_index=True)
        parsed_union['name_N_hw_comb'] = parsed_union.apply(unique_names, axis=1)

        # phase 1: Extract the unique name_N_hw_comb s.
        unique_name_N_hw_comb = parsed_union['name_N_hw_comb'].unique()

        # phase 2: For each unique name_N_hw_comb, find the median runtime.
        medians_runtimes = {}
        for name_N_hw_comb in unique_name_N_hw_comb:
            # get the rows corresponding to this unique name_N_hw_comb
            rows = parsed_union[parsed_union['name_N_hw_comb'] == name_N_hw_comb]
            median_runtime = rows['data_point'].median()
            conf = parse_unique_names(name_N_hw_comb)

            if conf['name'] not in medians_runtimes:
                medians_runtimes[conf['name']] = {}
            if conf['hw'] not in medians_runtimes[conf['name']]:
                medians_runtimes[conf['name']][conf['hw']] = {}
            if conf['N'] not in medians_runtimes[conf['name']][conf['hw']]:
                medians_runtimes[conf['name']][conf['hw']][conf['N']] = (conf, median_runtime)

        # phase 3: Find the best unroll factor for each N.
        for name, hw_dict in medians_runtimes.items():
            for hw, N_dict in hw_dict.items():
                for N, (conf, median_runtime) in N_dict.items():
                    filtered_conf = {k: v for k, v in conf.items() if k not in ['name', 'hw', 'N']}
                    print(f'Best configuration for {name}, on {hw}, for N={N} is {filtered_conf}')


if __name__ == '__main__':
    # Can be used as a standalone script
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps-dir', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--benchid', type=str, required=True)
    args = parser.parse_args()
    get_best_config(args.dumps_dir, args.benchid)



