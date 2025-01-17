import json
import os
import argparse
import pathlib
import numpy as np
import pandas as pd
import copy

from utils import *
from timer_stats import *
from lamda_funcs import *


class DumpsParser:
    def __init__(self, dumps_dirs_list: [str]):
        self.dumps_dirs_list = dumps_dirs_list
        self.dataframe_merged = pd.DataFrame()
        self.autotuner_merged = {}

    def parse_all(self):
        """
        Parse all the dumps directories in the list and merge the results.
        Returns nothing.
        """
        for dumps_dirs_index in range(len(self.dumps_dirs_list)):
            results = self._parse_one(dumps_dirs_index)
            parsed_dumps = self._convert_nested_dict_of_dataframes_to_dataframe(results[0])
            self.dataframe_merged = self._dataframes_concat(self.dataframe_merged, parsed_dumps)
            self.autotuner_merged = self._autotuner_merge_jsons(self.autotuner_merged, results[1])

    def get_dataframe_merged(self):
        """
        Get the merged data as a single DataFrame.
        """
        return self.dataframe_merged

    def get_autotuner_merged(self):
        """
        Get the merged autotuner JSON.
        """
        return self.autotuner_merged

    def get_autotuner_config(self, benchid: int, hw: str, best_run: bool):
        """
        Get the autotuner configuration for a specific benchmark and hardware.
        Returns either the config dict or None if not found.
        """
        benchid_str = ('0' if benchid < 10 else '') + str(benchid)
        if benchid_str not in self.autotuner_merged:
            return None
        if hw not in self.autotuner_merged[benchid_str]:
            return None
        if 'best' not in self.autotuner_merged[benchid_str][hw] and best_run:
            return None
        if 'auto-tune' not in self.autotuner_merged[benchid_str][hw] and not best_run:
            return None

        return self.autotuner_merged[benchid_str][hw]['best' if best_run else 'auto-tune']

    def _parse_one(self, dumps_dirs_index: int):
        """
        Parse a single dumps directory and load/parse all the subdirectories.
        returns a tuple: (a nested dictionary of DataFrames or None, autotuner json dict)
        """
        dump_path = self.dumps_dirs_list[dumps_dirs_index]

        # Check if the path exists and is a directory
        if not os.path.exists(dump_path):
            print(f'Error: {dump_path} does not exist')
            return None

        # Check if file `autotuner.json` in dump_path exists
        autotuner_path = os.path.join(dump_path, 'autotuner.json')
        if not os.path.exists(autotuner_path):
            print(f'Error: {autotuner_path} does not exist')
            return None
        else:
            print(f'Parsing {autotuner_path}')
            autotuner_cfg = json.load(open(autotuner_path))
            #print(autotuner_cfg)

        # Glob get all the .txt files in the root of the dump_path
        txt_files = pathlib.Path(dump_path).glob('*.txt')

        parsed_dumps = {}

        for txt_file in txt_files:
            print(f'Parsing {txt_file}')

            # If substr exists, then it's the runs for the auto-tuning phase
            # Otherwise, it's the runs with the best config found for benchmarking
            is_autotune_runs = '_autotune' in txt_file.name

            # Extract the benchmark Id from the file name:
            # benchId00.txt -> 00
            # benchId00_autotune.txt -> 00
            if is_autotune_runs:
                bench_id = txt_file.name.split('_')[0].split('benchId')[1]
            else:
                bench_id = txt_file.name.split('_')[0].split('benchId')[1]
                bench_id = bench_id[:bench_id.rfind('.')]
            bench_id_str = bench_id + ('_autotune' if is_autotune_runs else '')
            bench_id = int(bench_id)

            hw_names = get_all_hw_names(dump_path, bench_id_str)

            if bench_id not in parsed_dumps:
                parsed_dumps[bench_id] = {}
            for hw in hw_names:
                bench_id_str_all_json_paths = get_all_json_files(dump_path, bench_id_str, only_this_hw=hw)
                for json_path in bench_id_str_all_json_paths:
                    timer_stats_parser = TimerStatsParser(json_path, get_lambda_pairs(str(bench_id)))

                    if hw not in parsed_dumps[bench_id]:
                        parsed_dumps[bench_id][hw] = {}

                    parsed_dumps[bench_id][hw][
                        'auto-tune' if is_autotune_runs else 'best'] = timer_stats_parser.get_df()

        with open(pathlib.Path(self.dumps_dirs_list[dumps_dirs_index])/'autotuner.json', 'r') as f:
            parsed_json = json.load(f)

        return parsed_dumps, parsed_json

    def _convert_nested_dict_of_dataframes_to_dataframe(self, data):
        """
        Merge all pandas DataFrames in a nested dictionary structure into a single DataFrame.

        Parameters:
            data (dict): Nested dictionary of DataFrames.

        Returns:
            pd.DataFrame: Merged DataFrame with additional columns for the keys.
        """
        merged_frames = []

        for int_key, str_dict in data.items():
            for str_key1, str_dict2 in str_dict.items():
                for str_key2, df in str_dict2.items():
                    # Add columns for the keys
                    df = df.copy()  # Avoid modifying the original data frame
                    df['benchId'] = int_key
                    df['hw'] = str_key1
                    df['run_type'] = str_key2

                    # Append to the list of DataFrames
                    merged_frames.append(df)

        # Concatenate all DataFrames into one
        if merged_frames:
            merged_df = pd.concat(merged_frames, ignore_index=True)
        else:
            merged_df = pd.DataFrame()  # Return an empty DataFrame if no data

        return merged_df

    def _dataframes_concat(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """
        Merge two DataFrames into one.

        Parameters:
            df1 (pd.DataFrame): The first DataFrame.
            df2 (pd.DataFrame): The second DataFrame.

        Returns:
            pd.DataFrame: Merged DataFrame containing rows from both inputs.
        """
        if df1.empty:
            return df2.copy()
        if df2.empty:
            return df1.copy()

        return pd.concat([df1, df2], ignore_index=True)

    def _autotuner_merge_jsons(self, current_json_dict, new_json_dict):
        """
        Merge two JSON dictionaries into one, handling all the conflicts.
        Note that the arguments are not modified.
        """
        def deep_update_nested_dicts(original, updates):
            for key, value in updates.items():
                if key in original:
                    if isinstance(value, dict) and isinstance(original[key], dict):
                        # Recursively merge nested dictionaries
                        deep_update_nested_dicts(original[key], value)
                    elif original[key] != value:
                        # Raise exception on conflicting values
                        print(f'Error: conflicting values for key {key}')
                else:
                    # Add new key-value pairs
                    original[key] = value
            return original

        if not current_json_dict:
            return copy.deepcopy(new_json_dict)
        else:
            dst = copy.deepcopy(current_json_dict)
            return deep_update_nested_dicts(dst, new_json_dict)


if __name__ == '__main__':
    # accept multiple instances of --dumps arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps', type=str, required=True, nargs='+')
    args = parser.parse_args()
    dumps = args.dumps

    dumps_parser = DumpsParser(dumps)
    dumps_parser.parse_all()
    data = dumps_parser.get_dataframe_merged()

    print(data)
    print(dumps_parser.get_autotuner_merged())
    print(dumps_parser.get_autotuner_config(8, 'furore6', True))
