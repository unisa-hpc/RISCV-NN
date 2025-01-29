import argparse
import copy
import pandas as pd
import sys
from colorama import init, Fore
import json
import pathlib
from parsing.timer_stats import TimerStatsParser
from parsing.utils import *


def get_best_config(dumps_dir: str, benchid: int, out: str, parse_pairs_func=lambda pairs: {},
                    unique_names_two_args_func=lambda pairs, hw: ''):
    """
    Find the best unroll factor for each hardware and print it.
    It looks for benchId{benchid}.txt in the dumps_dir to extract all the relevant json files.

    Based on whether `-DAUTOTUNE_BASELINE_KERNELS` is defined or not, the baseline kernels in each benchmark
    will or will not be participated in the autotuning process.

    """
    init()
    all_hw_names = get_all_hw_names(dumps_dir, benchid, only_best=False)
    all_compiler_names = get_all_compiler_names(dumps_dir, benchid, only_best=False)

    # First, read existing JSON if it exists
    existing_json_report = {}
    if pathlib.Path(out).exists():
        print(f'Reading existing json report from {out}')
        with open(out, 'r') as f:
            existing_json_report = json.load(f)

    # Create new report dictionary
    json_report_dict = {}

    for hw_name in all_hw_names:
        for compiler_name in all_compiler_names:
            print('=================================================')
            print(
                f'Finding the best configuration for {Fore.GREEN}benchId{benchid}{Fore.RESET}, on {Fore.GREEN}{hw_name}{Fore.RESET}, with {Fore.GREEN}{compiler_name}{Fore.RESET} compiler')

            unique_names_func = lambda x: unique_names_two_args_func(x, hw_name)

            all_jsons = get_all_json_files(dumps_dir, benchid, hw_name, compiler_name, only_best=False)
            if len(all_jsons) == 0:
                print(f'No json files found for {benchid} on {hw_name}. Skipping...')
                continue

            parse_unique_names = lambda name: {
                pair.split('=')[0]: pair.split('=')[1] for pair in name.split(';;')
            }

            parsed_runs = [TimerStatsParser(j, parse_pairs_func) for j in all_jsons]
            parsed_union = pd.concat([run.get_df() for run in parsed_runs], ignore_index=True)
            parsed_union['name_N_hw_comb'] = parsed_union.apply(unique_names_func, axis=1)

            unique_name_N_hw_comb = parsed_union['name_N_hw_comb'].unique()

            configs_and_runtimes = {}
            for name_N_hw_comb in unique_name_N_hw_comb:
                rows = parsed_union[parsed_union['name_N_hw_comb'] == name_N_hw_comb]
                median_runtime = rows['data_point'].median()
                conf = parse_unique_names(name_N_hw_comb)

                key = (conf['name'], conf['hw'], conf['N'])
                if key not in configs_and_runtimes:
                    configs_and_runtimes[key] = []

                filtered_conf = copy.copy(conf)
                filtered_conf.pop('hw')
                filtered_conf.pop('N')
                filtered_conf.pop('name')

                configs_and_runtimes[key].append((filtered_conf, median_runtime))

            for (name, hw, N), configurations in configs_and_runtimes.items():
                sorted_configs = sorted(configurations, key=lambda x: x[1])
                best_config, best_runtime = sorted_configs[0]

                # Create nested structure if it doesn't exist
                if str(benchid) not in json_report_dict:
                    json_report_dict[str(benchid)] = {}
                if hw not in json_report_dict[str(benchid)]:
                    json_report_dict[str(benchid)][hw] = {}
                if compiler_name not in json_report_dict[str(benchid)][hw]:
                    json_report_dict[str(benchid)][hw][compiler_name] = {}

                json_report_dict[str(benchid)][hw][compiler_name][str(N)] = best_config

    # Merge with existing data
    for bench_id, hw_dict in json_report_dict.items():
        if bench_id not in existing_json_report:
            existing_json_report[bench_id] = {}

        for hw, compilers_dict in hw_dict.items():
            if hw not in existing_json_report[bench_id]:
                existing_json_report[bench_id][hw] = {}

            for compiler_name, N_dict in compilers_dict.items():
                if compiler_name not in existing_json_report[bench_id][hw]:
                    existing_json_report[bench_id][hw][compiler_name] = {}

                # Merge N_dict into existing data
                existing_json_report[bench_id][hw][compiler_name].update(N_dict)

    # Write merged result back to file
    with open(out, 'w') as f:
        json.dump(existing_json_report, indent=4, fp=f)


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
           'UNROLL_FACTOR0': pairs['UNROLL_FACTOR0']
        },
        lambda pairs, hw_name: \
           'name=' + pairs['name'] + f';;hw={hw_name}' + ';;N=' + str(pairs['N']) + ';;UNROLL_FACTOR0=' + str(
               pairs['UNROLL_FACTOR0']
        )
    )