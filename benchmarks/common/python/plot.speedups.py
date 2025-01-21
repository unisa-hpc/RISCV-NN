import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from parsing.parse import DumpsParser
from parsing.codebook import *
from parsing.lamda_funcs import *

import matplotlib as mpl


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
        self.dumps_parser = DumpsParser(dumps_dirs_list)
        self.raw_data = None
        self.proc_data = None
        self.proc_data_speedup = None
        self.dir_out = dir_out
        sns.set_theme(style="whitegrid")
        sns.color_palette("hls", 8)

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

        """
        The problem with speedups is that we CANNOT add them as new columns. We can have a speedup_type column.
        These will be added as rows to the speedup_type column:
        - Speedup_vv: Vectorized / Vectorized: avx2 base with avx2 uut, avx512 base with avx512 uut
        - Speedup_vs: Vectorized / Scalar: scalar no autovec base with avx2 uut, scalar no autovec base with avx512 uut
        - Speedup_ss: Scalar / Scalar: scalar no autovec base with scalar autovec base
        """
        unique_bids = self.proc_data['benchId'].unique()
        for bench_id in unique_bids:
            if bench_id == 0 or bench_id == 2 or bench_id == 3:
                print(f"NYI: Preprocessing data for benchID={bench_id}")
            elif bench_id == 7 or bench_id == 8:
                print(f"Preprocessing data for benchID={bench_id}")
                cols = list(self.proc_data.columns)
                cols.append('speedup_type')
                self.data_proc = pd.DataFrame(columns=cols)
                unique_bid_hw = self.proc_data['benchId_hw'].unique()
                unique_Ns = self.proc_data['N'].unique()
                # speedup_ss
                for bid_hw in unique_bid_hw:
                    sav_rows = self.proc_data.loc[
                        (self.proc_data['benchId_hw'] == bid_hw) &
                        (self.proc_data['name'].str.contains("SAV"))
                    ]
                    sna_rows = self.proc_data.loc[
                        (self.proc_data['benchId_hw'] == bid_hw) &
                        (self.proc_data['name'].str.contains("SNA"))
                    ]
                    sav_rows.reset_index(drop=True, inplace=True)
                    sna_rows.reset_index(drop=True, inplace=True)
                    # assert that the num of rows is the same
                    assert sav_rows.shape[0] == sna_rows.shape[0]
                    sav_rows.loc[:, 'data_point'] = sna_rows['data_point'] / sav_rows['data_point']  # speed up is ()^-1
                    sav_rows.loc[:, 'speedup_type'] = 'speedup_ss'
                    self.proc_data_speedup = pd.concat([self.proc_data_speedup, sav_rows], ignore_index=True)

                # speedup_vs
                # for speedup_vs, since our samples for scalar and vector kernels are not equal,
                # we need to reduce them manually and then calculate the speedup.
                # Basically we either:
                # [*] reduce everything to one sample with median operator and divide.
                # [*] reduce only the smaller group to one sample with median operator and divide every raw data entry in the larger group by the reduced val.
                # [ ] reduce everything to stats (min, max, median, ave) and divide the stats tuples and plot manually.
                # Since we are reducing, we have to mask everything down to the last combination (N, hw, name, configs, etc.)
                for bid_hw in unique_bid_hw:
                    for unique_n in unique_Ns:
                        avx2_rows = self.proc_data.loc[
                            (self.proc_data['benchId_hw'] == bid_hw) &
                            (self.proc_data['N'] == unique_n) &
                            (self.proc_data['run_type'] == 'best') &
                            (self.proc_data['name'].str.contains("AVX2"))
                        ]
                        avx512_rows = self.proc_data.loc[
                            (self.proc_data['benchId_hw'] == bid_hw) &
                            (self.proc_data['N'] == unique_n) &
                            (self.proc_data['run_type'] == 'best') &
                            (self.proc_data['name'].str.contains("AVX512"))
                        ]
                        sna_rows = self.proc_data.loc[
                            (self.proc_data['benchId_hw'] == bid_hw) &
                            (self.proc_data['N'] == unique_n) &
                            (self.proc_data['run_type'] == 'best') &
                            (self.proc_data['name'].str.contains("SNA"))
                        ]

                        avx2_rows.reset_index(drop=True, inplace=True)
                        avx512_rows.reset_index(drop=True, inplace=True)
                        sna_rows.reset_index(drop=True, inplace=True)

                        if sna_rows.empty: #TODO: benchId07 is broken here, FIX IT
                            print(f"Skipping N={unique_n} for {bid_hw} due to missing data sna_rows.")
                            continue

                        # reduce the smallest group to one sample, scalar kernels have no auto-tuning, only (hw, N, benchId)
                        sna_rows.loc[:, 'data_point'] = sna_rows['data_point'].median()
                        # only keep the first row of sna
                        sna_rows = sna_rows.iloc[[0]]

                        # some benchmarks only have avx512 or avx2 data
                        if avx2_rows.empty:
                            print(f"Skipping N={unique_n} for {bid_hw} due to missing data avx2_rows.")
                            continue
                        else:
                            avx2_rows['data_point'] = sna_rows['data_point'].iloc[0] / avx2_rows['data_point']
                            avx2_rows = avx2_rows.assign(speedup_type='speedup_vs')
                            self.proc_data_speedup = pd.concat([self.proc_data_speedup, avx2_rows], ignore_index=True)

                        if avx512_rows.empty:
                            print(f"Skipping N={unique_n} for {bid_hw} due to missing data avx512_rows.")
                            continue
                        else:
                            avx512_rows['data_point'] = sna_rows['data_point'].iloc[0] / avx512_rows['data_point']
                            avx512_rows = avx512_rows.assign(speedup_type='speedup_vs')
                            self.proc_data_speedup = pd.concat([self.proc_data_speedup, avx512_rows], ignore_index=True)

            elif bench_id == 1 or bench_id == 5 or bench_id == 6:
                print(f"Preprocessing data for benchID={bench_id}")
            elif bench_id == 4:
                print(f"NYI: Preprocessing data for benchID={bench_id}")
            else:
                print(f"Undefined benchID: {bench_id} for preprocessing.")

        self.proc_data_speedup['benchId_hw_name_speeduptype'] = \
            self.proc_data_speedup['benchId'].astype(str) + ';;' + \
            self.proc_data_speedup['hw'] + ';;' + \
            translate_codename_to(self.proc_data_speedup['name']) + ';;' + \
            self.proc_data_speedup['speedup_type']

        self.proc_data_speedup['benchId_hw_speeduptype'] = \
            self.proc_data_speedup['benchId'].astype(str) + ';;' + \
            self.proc_data_speedup['hw'] + ';;' + \
            self.proc_data_speedup['speedup_type']


    def plotgen_runtimes_all(self):
        """
        Generate all the plots, as many as needed for the parsed data.
        """
        self.preprocess_data()

        unique_n_list = self.proc_data['N'].unique()
        for n in unique_n_list:
            self.plotgen_runtimes_one(n)

    def plotgen_runtimes_one(self, n: int):
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
        order = sorted(masked_data['benchId_hw_name'].unique())

        sns.barplot(
            data=masked_data,
            x='benchId_hw_name',
            y='data_point',
            hue='benchId_hw',
            order=order,
            dodge=False,
            ci=95,  # Show 95% confidence intervals
            capsize=0.05  # Add caps to the error bars
        )
        # Customize the plot
        plt.title(f"Runtimes for N={n}")
        plt.xlabel("Group")
        plt.xticks(rotation=90)
        plt.ylabel("Runtime (ms)")
        plt.legend(title="Name", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Show the plot
        plt.show()

    def plotgen_speedups_all(self):
        """
        Generate all the plots, as many as needed for the parsed data.
        """
        self.preprocess_data()

        unique_n_list = self.proc_data_speedup['N'].unique()
        for n in unique_n_list:
            self.plotgen_speedups_one(n)

    def plotgen_speedups_one(self, n: int):
        # Extract the rows that have the specific N
        masked_data = self.proc_data_speedup[
            (self.proc_data_speedup['N'] == n)
            & (self.proc_data_speedup['benchId'] == 8)
        ]
        plt.figure(figsize=(12, 6))

        order = sorted(masked_data['benchId_hw_name_speeduptype'].unique())

        sns.barplot(
            data=masked_data,
            x='benchId_hw_name_speeduptype',
            y='data_point',
            hue='benchId_hw',
            palette='viridis',
            order=order,
            dodge=False, # Do not set this to true. It will cause offset to the bars.
            ci=95,  # Show 95% confidence intervals
            capsize=0.05  # Add caps to the error bars
        )
        # Customize the plot
        plt.title(f"Speedup for N={n}")
        plt.xlabel("Group")
        plt.xticks(rotation=90)
        plt.ylabel("Speedup")
        plt.legend(title="Name", bbox_to_anchor=(1.05, 1), loc='upper left')
        #plt.tight_layout()
        plt.subplots_adjust(bottom=0.5)  # Adjust the bottom margin

        # Show the plot
        plt.show()

if __name__ == '__main__':
    # accept multiple instances of --dumps arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps', type=str, required=True, action='append')
    args = parser.parse_args()
    dumps = args.dumps

    obj = PlotSpeedUps(dumps, '/tmp')
    obj.plotgen_runtimes_all()
    obj.plotgen_speedups_all()