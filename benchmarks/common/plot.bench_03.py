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


def get_all_json_files(dump_dir: str, bench_id: str) -> [str]:
    """
    Get the abs path of all the json files in the dump directory, recursively.
    """
    # read all the lines from the file at dump_dir/benchId.txt
    with open(pathlib.Path(dump_dir).joinpath(f'benchId{bench_id}.txt'), 'r') as f:
        lines = f.readlines()

    json_files = []
    for current_run_dir in lines:
        # find all the json files in current_run_dir
        current_run_dir = current_run_dir.strip()
        current_run_dir_path = pathlib.Path(current_run_dir)
        json_files.extend([str(f) for f in current_run_dir_path.rglob('*.json')])

    print(f'Found {len(lines)} sub-dump directories and a '
          f'total of {len(json_files)} json files for benchmark ID {bench_id}')
    return json_files


def _plot(data_constant_config: pd.DataFrame, idx, col_name, col_value, sweep_col_name, desc_dict: dict):
    plt.clf()  # Clear the current figure
    f = data_constant_config[col_name] == col_value
    masked_data = data_constant_config[f]
    sorted_unique_names = sorted(masked_data['name'].unique())

    legend_handles = []
    ax = plt.gca()  # Get the current Axes instance
    for i, current_name in enumerate(sorted_unique_names):
        line_data = masked_data[masked_data['name'] == current_name]
        sns.lineplot(
            data=line_data,
            x=sweep_col_name,
            y='data_point',
            ax=ax,
            legend=False
        )

    for line, current_name in zip(ax.get_lines(), sorted_unique_names):
        legend_handles.append(Line2D([0], [0], color=line.get_color(), label=current_name))

    plt.title(f'Config for {desc_dict[col_name]} =\n{col_name}={col_value}, sweep={sweep_col_name}', fontsize=8)
    plt.xlabel(sweep_col_name)
    plt.ylabel('Time (ms)')
    plt.legend(handles=legend_handles, title='Config', fontsize=8, title_fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout
    plt.subplots_adjust(right=0.6, hspace=0.2)
    plt.gcf().set_size_inches(10, 6)  # Adjust the canvas size
    plt.savefig(pathlib.Path(args.dumps_dir).joinpath(args.out).joinpath(f'{col_name}_{col_value}_{sweep_col_name}.png'))

if __name__ == '__main__':
    # add -d argument for dumps dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps-dir', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--benchid', type=str, required=True)
    args = parser.parse_args()

    all_jsons = get_all_json_files(args.dumps_dir, args.benchid)
    parse_pairs = lambda pairs: {
        'I_H': pairs['I_H'],
        'I_W': pairs['I_W'],
        'K_H': pairs['K_H'],
        'K_W': pairs['K_W'],
        'C_I': pairs['C_I'],
        'C_O': pairs['C_O'],
        'S_X': pairs['S_X'],
        'S_Y': pairs['S_Y'],
        'UNROLL_FACTOR0': pairs['UNROLL_FACTOR0'],
        'UNROLL_FACTOR1': pairs['UNROLL_FACTOR1'],
        'UNROLL_FACTOR2': pairs['UNROLL_FACTOR2'],
        'UNROLL_FACTOR3': pairs['UNROLL_FACTOR3']
    }
    parsed_runs = [TimerStatsParser(j, parse_pairs) for j in all_jsons]
    parsed_union = pd.concat([run.get_df() for run in parsed_runs], ignore_index=True)
    legend_desc = {}
    # combine all pairs to form a unique config identifier # parsed_union['name'] + '-' + \
    parsed_union['config_id'] = \
        parsed_union['I_H'].astype(str) + '-' + parsed_union['I_W'].astype(str) + '-' + \
        parsed_union['K_H'].astype(str) + '-' + parsed_union['K_W'].astype(str) + '-' + \
        parsed_union['C_I'].astype(str) + '-' + parsed_union['C_O'].astype(str) + '-' + \
        parsed_union['S_X'].astype(str) + '-' + parsed_union['S_Y'].astype(str)

    legend_desc['config_id'] = 'I_H, I_W, K_H, K_W, C_I, C_O, S_X, S_Y'

    parsed_union['unique_factors123'] = parsed_union['UNROLL_FACTOR1'].astype(str) + '-' + \
                                        parsed_union['UNROLL_FACTOR2'].astype(str) + '-' + \
                                        parsed_union['UNROLL_FACTOR3'].astype(str)
    parsed_union['unique_factors023'] = parsed_union['UNROLL_FACTOR0'].astype(str) + '-' + \
                                        parsed_union['UNROLL_FACTOR2'].astype(str) + '-' + \
                                        parsed_union['UNROLL_FACTOR3'].astype(str)
    parsed_union['unique_factors013'] = parsed_union['UNROLL_FACTOR0'].astype(str) + '-' + \
                                        parsed_union['UNROLL_FACTOR1'].astype(str) + '-' + \
                                        parsed_union['UNROLL_FACTOR3'].astype(str)
    parsed_union['unique_factors012'] = parsed_union['UNROLL_FACTOR0'].astype(str) + '-' + \
                                        parsed_union['UNROLL_FACTOR1'].astype(str) + '-' + \
                                        parsed_union['UNROLL_FACTOR2'].astype(str)

    unique_configs = parsed_union['config_id'].unique()

    # Add a new column for each sweep factor
    sweep_index_to_unique_factors = {0: 'unique_factors123', 1: 'unique_factors023', 2: 'unique_factors013', 3: 'unique_factors012'}
    for cfg in unique_configs:
        for sweep_factor in range(4):
            # create a new column by combining config_id and sweep_index_to_unique_factors[sweep_factor]
            parsed_union[f'config_id_sweep_factor{sweep_factor}'] = parsed_union['config_id'] + ' (' + parsed_union[sweep_index_to_unique_factors[sweep_factor]] + ')'
            legend_desc[f'config_id_sweep_factor{sweep_factor}'] = f'{legend_desc["config_id"]} ({sweep_index_to_unique_factors[sweep_factor]})'

    for sweep_factor in range(4):
        unique_configs_for_current_sweep_factor = parsed_union[f'config_id_sweep_factor{sweep_factor}'].unique()
        for idx, col_value_unique in enumerate(unique_configs_for_current_sweep_factor):
            # plot each case
            _plot(parsed_union, idx, f'config_id_sweep_factor{sweep_factor}', col_value_unique, f'UNROLL_FACTOR{sweep_factor}', legend_desc)




