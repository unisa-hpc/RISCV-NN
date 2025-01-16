from matplotlib import pyplot as plt
import argparse
import json
import pathlib
import pandas as pd
import seaborn as sns
import sys
from matplotlib.lines import Line2D

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2].joinpath('common/python')))
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


if __name__ == '__main__':
    # add -d argument for dumps dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps-dir', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--benchid', type=str, required=True)
    args = parser.parse_args()

    all_jsons = get_all_json_files(args.dumps_dir, args.benchid)
    parse_pairs = lambda pairs: {
        'N': pairs['N'],
        'unroll_factor': pairs['unroll_factor']
    }
    parsed_runs = [TimerStatsParser(j, parse_pairs) for j in all_jsons]
    parsed_union = pd.concat([run.get_df() for run in parsed_runs], ignore_index=True)
    parsed_union['name_N'] = parsed_union['name'] + ' (N=' + parsed_union['N'].astype(str) + ')'

    # Define unique markers and dash patterns for each unique "N"
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    dash_styles = [(2, 1), (4, 1), (6, 2), (8, 2), (3, 1, 1, 1), (5, 2, 1, 2)]
    unique_N_values = parsed_union['N'].unique()

    marker_dict = {N: markers[i % len(markers)] for i, N in enumerate(unique_N_values)}
    dash_dict = {N: dash_styles[i % len(dash_styles)] for i, N in enumerate(unique_N_values)}

    # Calculate subplot layout
    num_subplots = len(unique_N_values)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 6 * num_subplots), sharex=True)
    if num_subplots == 1:
        axes = [axes]  # Ensure axes is always a list for consistent indexing

    # Sort names for each N group and create a color palette
    color_palettes = {}
    for N in unique_N_values:
        sorted_names = sorted(parsed_union[parsed_union['N'] == N]['name_N'].unique())
        color_palettes[N] = sns.color_palette("hsv", len(sorted_names))

    for idx, N in enumerate(unique_N_values):
        ax = axes[idx]
        data_N = parsed_union[parsed_union['N'] == N]

        # Sort the unique names for color consistency
        sorted_name_N = sorted(data_N['name_N'].unique())

        # Plot each line in the subplot for this specific N, using sorted colors, custom marker, and dashes
        for i, name_N in enumerate(sorted_name_N):
            line_data = data_N[data_N['name_N'] == name_N]
            sns.lineplot(
                data=line_data,
                x='unroll_factor',
                y='data_point',
                marker=marker_dict[N],
                dashes=dash_dict[N],
                ax=ax,
                color=color_palettes[N][i],  # Assign sorted color
                legend=False
            )

        # ax.set_yscale('log')
        ax.set_title(f"Runtimes vs. Unrolling Factors for N={N}")
        ax.set_xlabel("Unroll Factor")
        ax.set_ylabel("Runtime (ms)")

        # Manually create the legend with only relevant entries for this subplot (N)
        legend_elements = [
            Line2D([0], [0], color=color_palettes[N][i],  # Use sorted color for this N
                   marker=marker_dict[N],
                   dashes=dash_dict[N],
                   label=name_N)
            for i, name_N in enumerate(sorted_name_N)
        ]
        ax.legend(handles=legend_elements, title="Name", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout
    plt.subplots_adjust(right=0.6, hspace=0.2)
    plt.savefig(pathlib.Path(args.dumps_dir).joinpath(args.out))
