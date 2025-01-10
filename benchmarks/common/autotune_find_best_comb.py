import argparse
import copy
import pandas as pd
import sys
from colorama import init, Fore
import json
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2].joinpath('common')))
from timer_stats import TimerStatsParser
from utils import *


def get_best_config(dumps_dir: str, benchid: str, out: str, parse_pairs_func=lambda pairs: {}, unique_names_two_args_func=lambda pairs, hw: ''):
    """
    Find the best unroll factor for each hardware and print it.
    It looks for benchId{benchid}.txt in the dumps_dir to extract all the relevant json files.
    """
    init()
    all_hw_names = get_all_hw_names(dumps_dir, benchid)

    # json_report structure: (no lists, everything is a dict)
    # benchId -> hw_name -> N -> config -> {key: value}
    json_report_dict = {}

    for hw_name in all_hw_names:
        print('=================================================')
        print('Finding best unroll factor for hardware:', hw_name)

        unique_names_func = lambda x: unique_names_two_args_func(x, hw_name)

        all_jsons = get_all_json_files(dumps_dir, benchid, hw_name)
        parse_unique_names = lambda name: {
            pair.split('=')[0]: pair.split('=')[1] for pair in name.split(';;')
        }

        parsed_runs = [TimerStatsParser(j, parse_pairs_func) for j in all_jsons]
        parsed_union = pd.concat([run.get_df() for run in parsed_runs], ignore_index=True)
        parsed_union['name_N_hw_comb'] = parsed_union.apply(unique_names_func, axis=1)

        # phase 1: Extract the unique name_N_hw_comb s.
        unique_name_N_hw_comb = parsed_union['name_N_hw_comb'].unique()

        # phase 2: For each unique name_N_hw_comb, collect all configurations and their runtimes
        configs_and_runtimes = {}
        for name_N_hw_comb in unique_name_N_hw_comb:
            # get the rows corresponding to this unique name_N_hw_comb
            rows = parsed_union[parsed_union['name_N_hw_comb'] == name_N_hw_comb]
            median_runtime = rows['data_point'].median()
            conf = parse_unique_names(name_N_hw_comb)

            key = (conf['name'], conf['hw'], conf['N'])
            if key not in configs_and_runtimes:
                configs_and_runtimes[key] = []

            filtered_conf = copy.copy(conf)

            # Filter out hw, N, benchId and keep the rest
            filtered_conf.pop('hw')
            filtered_conf.pop('N')
            filtered_conf.pop('name')

            configs_and_runtimes[key].append((filtered_conf, median_runtime))

        # phase 3: Find and print the best unroll factor and all configurations for each case
        for (name, hw, N), configurations in configs_and_runtimes.items():
            # Sort configurations by runtime
            sorted_configs = sorted(configurations, key=lambda x: x[1])
            best_config, best_runtime = sorted_configs[0]

            # Format all configurations for printing
            all_configs_str = ', '.join([
                f"{conf} (runtime: {runtime:.2f})"
                for conf, runtime in sorted_configs
            ])

            msg = (f'Best configuration for {Fore.GREEN}benchId{benchid}{Fore.RESET}, '
                   f'kernel ({Fore.GREEN}{name}{Fore.RESET}), on {Fore.GREEN}{hw}{Fore.RESET}, '
                   f'for N={Fore.GREEN}{N}{Fore.RESET} is {Fore.RED}{best_config}{Fore.RESET} '
                   f'FROM {Fore.YELLOW}[{all_configs_str}]{Fore.RESET}'
            )
            print(msg)

            # Create keys if they don't exist in json_report
            if benchid not in json_report_dict:
                json_report_dict[benchid] = {}
            if hw not in json_report_dict[benchid]:
                json_report_dict[benchid][hw] = {}
            if str(N) not in json_report_dict[benchid][hw]:
                json_report_dict[benchid][hw][str(N)] = {}
            json_report_dict[benchid][hw][str(N)] = best_config

    json_report = json.dumps(json_report_dict, indent=4)
    # check if out file exists, if exists, read it and merge with json_report_dict and write it back (overwrite)
    # if not exists, create it and write json_report_dict
    # if there is conflict, throw exception

    if pathlib.Path(out).exists():
        print(f'Merging the new json report with the existing one at {out}')
        with open(out, 'r') as f:
            existing_json_report = json.load(f)

        # merge the two dictionaries
        for bench_id, hw_dict in json_report_dict.items():
            if bench_id not in existing_json_report:
                existing_json_report[bench_id] = {}

            for hw, N_dict in hw_dict.items():
                if hw not in existing_json_report[bench_id]:
                    existing_json_report[bench_id][hw] = {}

                for N, config in N_dict.items():
                    if N not in existing_json_report[bench_id][hw]:
                        existing_json_report[bench_id][hw][N] = config
                    else:
                        # Check if the existing configuration matches the new one
                        existing_config = existing_json_report[bench_id][hw][N]
                        if existing_config != config:
                            raise Exception(
                                f"Conflict in the existing JSON file {out} for benchId {bench_id}, hw {hw}, N {N}.\n"
                                f"Existing: {existing_config}, New: {config}"
                            )
        # If no exception was raised, the dictionaries are merged successfully.

        # write the merged dictionary back to the file
        with open(out, 'w') as f:
            json.dump(existing_json_report, f, indent=4)
    else:
        print(f'Writing the new json report to {out}')
        with open(out, 'w') as f:
            f.write(json_report)


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