import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from parsing.parse import DumpsParser
from parsing.codebook import *


class PlotSpeedUps:
    """
    So the idea of this plot is to:
    1. Have only speedups as groups of bar graphs for each (benchId, hw) pair.
    2. The work-size is fixed for the entire plot (e.g. N=1024 would be one figure, N=2048 would be another figure, etc.)
    3. Included kernels:
            - Scalar Base no-autovec / no auto-tuned
            - Scalar Base autovec / no auto-tuned
            - Vector Base * ? / no auto-tuned
            - Vector UUT * ? / auto-tuned
            - Vector UUT * ? / default params for tunable params.
    """

    def __init__(self, dumps_dirs_list: [str], dir_out: str):
        self.parser = DumpsParser(dumps_dirs_list)
        self.dumps_parser = DumpsParser(dumps_dirs_list)
        self.raw_data = None
        self.proc_data = None
        self.dir_out = dir_out

    def load_data(self):
        self.dumps_parser.parse_all()
        self.raw_data = self.dumps_parser.get_dataframe_merged()
        self.proc_data = self.raw_data.copy()

    def preprocess_data(self):
        """
        Preprocess the raw data (self.raw_data) to get the speedups.

        {
            "Group": ["group_1"] * 3 + ["group_2"] * 4 + ["group_3"] * 2,
            "Bar Name": ["bar_1", "bar_2", "bar_3", "bar_4", "bar_5", "bar_6", "bar_7", "bar_8", "bar_9"],
            "Value": [5, 8, 3, 10, 7, 6, 2, 11, 9],  # Modified unique values for each bar
        }
        """
        if self.raw_data is None:
            self.load_data()
        self._preprocess_add_columns()


    def _preprocess_add_columns(self):
        """
        Add a column to the raw data that is a combination of the name and hw.
        """
        self.proc_data['name_hw'] = \
            self.proc_data['name'] + ';;' + \
            self.proc_data['hw']

        self.proc_data['benchId_hw_name'] = \
            self.proc_data['benchId'].astype(str)  + ';;' + \
            self.proc_data['hw'] + ';;' + \
            translate_codename_to(self.proc_data['name'])

        self.proc_data['benchId_hw'] = \
            self.proc_data['benchId'].astype(str) + ';;' + \
            self.proc_data['hw']

    def generate_all_plots(self):
        """
        Generate all the plots, as many as needed for the parsed data.
        """
        self.preprocess_data()

        unique_n_list = self.proc_data['N'].unique()
        for n in unique_n_list:
            self.generate_plot(n)

    def generate_plot(self, n: int):
        """
        Generate a plot for a specific N.
        """
        # Extract the rows that have the specific N
        masked_data = self.proc_data[self.proc_data['N'] == n]

        # Extract the group names, each group is a unique (benchId, hw) pair
        group_names = masked_data['benchId_hw'].unique()
        group_names = sorted(group_names)

        # So we need to create k-bars and assign a group for each one.
        # First thing to do is to find how many bars we need to create.
        # Each bar has a unique (name, hw, benchId) combination. N is already fixed. So we use col `benchId_hw_name`.
        unique_bars = masked_data['benchId_hw_name'].unique()
        unique_bars = sorted(unique_bars)

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=masked_data,
            x='benchId_hw',
            y='data_point',
            hue='benchId_hw_name',
            dodge=True,
            ci=95,  # Show 95% confidence intervals
            capsize=0.05  # Add caps to the error bars
        )
        # Customize the plot
        plt.title(f"Runtimes for N={n}")
        plt.xlabel("Group")
        plt.ylabel("Runtime (ms)")
        plt.legend(title="Name", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Show the plot
        plt.show()


        pass


if __name__ == '__main__':
    # accept multiple instances of --dumps arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps', type=str, required=True, nargs='+')
    args = parser.parse_args()
    dumps = args.dumps

    PlotSpeedUps(dumps, '/tmp').generate_all_plots()