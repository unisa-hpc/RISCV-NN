from matplotlib import pyplot as plt
import argparse
import json
import pathlib
import pandas as pd
import seaborn as sns
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2].joinpath('common')))
from timer_stats import TimerStatsParser

def get_all_json_files(dump_dir) -> [str]:
    """
    Get the abs path of all the json files in the dump directory, recursively.
    """
    return list(pathlib.Path(dump_dir).rglob('*.json'))

if __name__ == '__main__':
    # add -d argument for dumps dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps-dir', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    all_jsons = get_all_json_files(args.dumps_dir)
    parsed_runs = [TimerStatsParser(j) for j in all_jsons]
    parsed_union = pd.concat([run.get_df() for run in parsed_runs], ignore_index=True)
    parsed_union['name_N'] = parsed_union['name'] + ' (N=' + parsed_union['N'].astype(str) + ')'

    # Define unique markers and dash patterns for each unique "N"
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    dash_styles = [(2, 1), (4, 1), (6, 2), (8, 2), (3, 1, 1, 1), (5, 2, 1, 2)]
    unique_N_values = parsed_union['N'].unique()

    marker_dict = {N: markers[i % len(markers)] for i, N in enumerate(unique_N_values)}
    dash_dict = {N: dash_styles[i % len(dash_styles)] for i, N in enumerate(unique_N_values)}

    # Map markers and dash styles based on "N" values
    parsed_union['marker'] = parsed_union['N'].map(marker_dict)
    parsed_union['dashing'] = parsed_union['N'].map(dash_dict)

    # Plot with sns.lineplot, grouping colors by 'name_N' and styles by 'N', but suppressing the default legend
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=parsed_union,
        x='unroll_factor',
        y='data_point',
        hue='name_N',       # Use `name_N` for colors
        style='N',          # Use `N` for consistent markers and dashes across groups
        markers=marker_dict,
        dashes=dash_dict,
        legend=False        # Suppress default legend to avoid extra entries
    )
    ax.set_yscale('log')

    # Manually create the legend with the unique `name_N` entries
    from matplotlib.lines import Line2D
    legend_elements = []
    for name_N in parsed_union['name_N'].unique():
        N_value = parsed_union.loc[parsed_union['name_N'] == name_N, 'N'].iloc[0]
        legend_elements.append(Line2D(
            [0], [0],
            color=ax.get_lines()[len(legend_elements)].get_color(),
            marker=marker_dict[N_value],
            dashes=dash_dict[N_value],
            label=name_N
        ))

    # Add the custom legend
    ax.legend(handles=legend_elements, title="Name", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    # Add title and labels
    plt.title("Runtimes vs. Unrolling Factors")
    plt.xlabel("Unroll Factor")
    plt.ylabel("Runtime (ms)")

    # Adjust the right margin to make room for the legend
    plt.subplots_adjust(right=0.7)

    plt.savefig(pathlib.Path(args.dumps_dir).joinpath(args.out))
