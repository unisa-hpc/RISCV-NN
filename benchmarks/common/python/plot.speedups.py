import argparse
import inspect

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
        self.proc_data['name_hw_compiler'] = \
            self.proc_data['name'] + ';;' + \
            self.proc_data['hw'] + ';;' + \
            self.proc_data['compiler']

        self.proc_data['benchId_hw_compiler_name'] = \
            self.proc_data['benchId'].astype(str) + ';;' + \
            self.proc_data['hw'] + ';;' + \
            self.proc_data['compiler'] + ';;' + \
            translate_codename_to(self.proc_data['name'])

        self.proc_data['benchId_hw_compiler'] = \
            'BenchId' + self.proc_data['benchId'].astype(str) + ', ' + \
            self.proc_data['hw'] + ', ' + \
            self.proc_data['compiler']

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
                unique_bid_hw_compiler = self.proc_data['benchId_hw_compiler'].unique()
                unique_Ns = self.proc_data['N'].unique()
                # speedup_ss
                for bid_hw_compiler in unique_bid_hw_compiler:
                    sav_rows = self.proc_data.loc[
                        (self.proc_data['benchId_hw_compiler'] == bid_hw_compiler) &
                        (self.proc_data['name'].str.contains("SAV")) &
                        (self.proc_data['FLAG_AUTOTUNE_DISABLED'] == 0)
                        ]
                    sna_rows = self.proc_data.loc[
                        (self.proc_data['benchId_hw_compiler'] == bid_hw_compiler) &
                        (self.proc_data['name'].str.contains("SNA")) &
                        (self.proc_data['FLAG_AUTOTUNE_DISABLED'] == 0)
                        ]
                    if sav_rows.empty or sna_rows.empty:
                        print(f"Skipping {bid_hw_compiler} due to missing data sav_rows or sna_rows.")
                        continue
                    sav_rows.reset_index(drop=True, inplace=True)
                    sna_rows.reset_index(drop=True, inplace=True)
                    # assert that the num of rows is the same
                    assert sav_rows.shape[0] == sna_rows.shape[0]
                    sav_rows.loc[:, 'data_point'] = sna_rows['data_point'] / sav_rows['data_point']  # speed up is ()^-1
                    sav_rows.loc[:, 'speedup_type'] = 'speedup_ss'
                    self.proc_data_speedup = pd.concat([self.proc_data_speedup, sav_rows], ignore_index=True)

                # speedup_vs
                # for speedup_vs, since our samples for scalar and vector kernels are not equal,
                # So to keep it legit, for each sample for scalars kernels, we calculate N speedups_vs for vector kernels.
                # Since we are reducing, we have to mask everything down to the last combination (N, hw, name, configs, etc.)
                for bid_hw_compiler in unique_bid_hw_compiler:
                    for unique_n in unique_Ns:
                        avx2_rows = self.proc_data.loc[
                            (self.proc_data['benchId_hw_compiler'] == bid_hw_compiler) &
                            (self.proc_data['N'] == unique_n) &
                            (self.proc_data['run_type'] == 'best') &
                            (self.proc_data['name'].str.contains("AVX2")) &
                            (self.proc_data['FLAG_AUTOTUNE_DISABLED'] == 0)
                            ]
                        avx512_rows = self.proc_data.loc[
                            (self.proc_data['benchId_hw_compiler'] == bid_hw_compiler) &
                            (self.proc_data['N'] == unique_n) &
                            (self.proc_data['run_type'] == 'best') &
                            (self.proc_data['name'].str.contains("AVX512")) &
                            (self.proc_data['FLAG_AUTOTUNE_DISABLED'] == 0)
                            ]
                        sna_rows = self.proc_data.loc[
                            (self.proc_data['benchId_hw_compiler'] == bid_hw_compiler) &
                            (self.proc_data['N'] == unique_n) &
                            (self.proc_data['run_type'] == 'best') &
                            (self.proc_data['name'].str.contains("SNA")) &
                            (self.proc_data['FLAG_AUTOTUNE_DISABLED'] == 0)
                            ]

                        avx2_rows.reset_index(drop=True, inplace=True)
                        avx512_rows.reset_index(drop=True, inplace=True)
                        sna_rows.reset_index(drop=True, inplace=True)

                        if sna_rows.empty:
                            print(f"Skipping N={unique_n} for {bid_hw_compiler} due to missing data sna_rows.")
                            continue

                        # Create copies of the original dataframes to avoid modifying them in the loop
                        avx2_rows_original = avx2_rows.copy()
                        avx512_rows_original = avx512_rows.copy()

                        # loop over sna_rows['data_point'] and concat the speedup_vs rows
                        for sna_row in sna_rows['data_point']:
                            # Handle AVX2 calculations
                            if not avx2_rows.empty:
                                # Create a new copy for this iteration
                                avx2_rows_current = avx2_rows_original.copy()
                                avx2_rows_current['data_point'] = sna_row / avx2_rows_original['data_point']
                                avx2_rows_current = avx2_rows_current.assign(speedup_type='speedup_vs')
                                self.proc_data_speedup = pd.concat([self.proc_data_speedup, avx2_rows_current],
                                                                   ignore_index=True)
                            else:
                                print(f"Skipping N={unique_n} for {bid_hw_compiler} due to missing data avx2_rows.")

                            # Handle AVX512 calculations
                            if not avx512_rows.empty:
                                # Create a new copy for this iteration
                                avx512_rows_current = avx512_rows_original.copy()
                                avx512_rows_current['data_point'] = sna_row / avx512_rows_original['data_point']
                                avx512_rows_current = avx512_rows_current.assign(speedup_type='speedup_vs')
                                self.proc_data_speedup = pd.concat([self.proc_data_speedup, avx512_rows_current],
                                                                   ignore_index=True)
                            else:
                                print(f"Skipping N={unique_n} for {bid_hw_compiler} due to missing data avx512_rows.")

                # speedup_vv: avx2 base with avx2 uut, avx512 base with avx512 uut (all auto-tuned)
                for bid_hw_compiler in unique_bid_hw_compiler:
                    for unique_n in unique_Ns:
                        # BenchId 7 and 8 have only Baselines (avx2 and avx512) and Ours (avx512). So we only have 1 speedup VV for avx512.
                        avx512_ours_rows = self.proc_data.loc[
                            (self.proc_data['benchId_hw_compiler'] == bid_hw_compiler) &
                            (self.proc_data['N'] == unique_n) &
                            (self.proc_data['run_type'] == 'best') &
                            (self.proc_data['name'].str.contains("AVX512")) &
                            (self.proc_data['name'].str.contains("ours")) &
                            (self.proc_data['FLAG_AUTOTUNE_DISABLED'] == 0)
                            ]
                        avx512_baseline_rows = self.proc_data.loc[
                            (self.proc_data['benchId_hw_compiler'] == bid_hw_compiler) &
                            (self.proc_data['N'] == unique_n) &
                            (self.proc_data['run_type'] == 'best') &
                            (self.proc_data['name'].str.contains("AVX512")) &
                            (self.proc_data['name'].str.contains("base")) &
                            (self.proc_data['FLAG_AUTOTUNE_DISABLED'] == 0)
                            ]
                        if avx512_ours_rows.empty or avx512_baseline_rows.empty:
                            print(
                                f"Skipping N={unique_n} for {bid_hw_compiler} due to missing data avx512_ours_rows or avx512_baseline_rows.")
                            continue
                        avx512_ours_rows.reset_index(drop=True, inplace=True)
                        avx512_baseline_rows.reset_index(drop=True, inplace=True)
                        # assert that the num of rows is the same
                        assert avx512_ours_rows.shape[0] == avx512_baseline_rows.shape[0]
                        avx512_ours_rows.loc[:, 'data_point'] = avx512_baseline_rows['data_point'] / avx512_ours_rows[
                            'data_point']  # speed up is ()^-1
                        avx512_ours_rows.loc[:, 'speedup_type'] = 'speedup_vv'
                        self.proc_data_speedup = pd.concat([self.proc_data_speedup, avx512_ours_rows],
                                                           ignore_index=True)

                # auto-tuning gain
                for bid_hw_compiler in unique_bid_hw_compiler:
                    for unique_n in unique_Ns:
                        # BenchId 7 and 8 have only Baselines (avx2 and avx512) and Ours (avx512). So we only have 1 gain for avx512.
                        avx512_ours_rows = self.proc_data.loc[
                            (self.proc_data['benchId_hw_compiler'] == bid_hw_compiler) &
                            (self.proc_data['N'] == unique_n) &
                            (self.proc_data['run_type'] == 'best') &
                            (self.proc_data['name'].str.contains("AVX512")) &
                            (self.proc_data['name'].str.contains("ours")) &
                            (self.proc_data['FLAG_AUTOTUNE_DISABLED'] == 0)
                            ]
                        avx512_ours_no_autotune_rows = self.proc_data.loc[
                            (self.proc_data['benchId_hw_compiler'] == bid_hw_compiler) &
                            (self.proc_data['N'] == unique_n) &
                            (self.proc_data['run_type'] == 'best') &
                            (self.proc_data['name'].str.contains("AVX512")) &
                            (self.proc_data['name'].str.contains("ours")) &
                            (self.proc_data['FLAG_AUTOTUNE_DISABLED'] == 1)
                            ]
                        if avx512_ours_rows.empty or avx512_ours_no_autotune_rows.empty:
                            print(
                                f"Skipping N={unique_n} for {bid_hw_compiler} due to missing data avx512_ours_rows or avx512_ours_no_autotune_rows.")
                            continue
                        avx512_ours_rows.reset_index(drop=True, inplace=True)
                        avx512_ours_no_autotune_rows.reset_index(drop=True, inplace=True)
                        # assert that the num of rows is the same
                        assert avx512_ours_rows.shape[0] == avx512_ours_no_autotune_rows.shape[0]
                        avx512_ours_rows.loc[:, 'data_point'] = avx512_ours_no_autotune_rows['data_point'] / \
                                                                avx512_ours_rows[
                                                                    'data_point']  # speed up is ()^-1
                        avx512_ours_rows.loc[:, 'speedup_type'] = 'autotuning_gain'
                        self.proc_data_speedup = pd.concat([self.proc_data_speedup, avx512_ours_rows],
                                                           ignore_index=True)


            elif bench_id == 1 or bench_id == 5 or bench_id == 6:
                print(f"Preprocessing data for benchID={bench_id}")
                cols = list(self.proc_data.columns)
                cols.append('speedup_type')
                self.data_proc = pd.DataFrame(columns=cols)
                unique_bid_hw_compiler = self.proc_data['benchId_hw_compiler'].unique()
                unique_Ns = self.proc_data['N'].unique()
                # speedup_ss
                for bid_hw_compiler in unique_bid_hw_compiler:
                    sav_rows = self.proc_data.loc[
                        (self.proc_data['benchId_hw_compiler'] == bid_hw_compiler) &
                        (self.proc_data['name'].str.contains("SAV"))
                        ]
                    sna_rows = self.proc_data.loc[
                        (self.proc_data['benchId_hw_compiler'] == bid_hw_compiler) &
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
                for bid_hw_compiler in unique_bid_hw_compiler:
                    for unique_n in unique_Ns:
                        rvv_rows = self.proc_data.loc[
                            (self.proc_data['benchId_hw_compiler'] == bid_hw_compiler) &
                            (self.proc_data['N'] == unique_n) &
                            (self.proc_data['run_type'] == 'best') &
                            (self.proc_data['name'].str.contains("RVV"))
                            ]
                        sna_rows = self.proc_data.loc[
                            (self.proc_data['benchId_hw_compiler'] == bid_hw_compiler) &
                            (self.proc_data['N'] == unique_n) &
                            (self.proc_data['run_type'] == 'best') &
                            (self.proc_data['name'].str.contains("SNA"))
                            ]

                        rvv_rows.reset_index(drop=True, inplace=True)
                        sna_rows.reset_index(drop=True, inplace=True)

                        if sna_rows.empty:
                            print(f"Skipping N={unique_n} for {bid_hw_compiler} due to missing data sna_rows.")
                            continue

                        # loop over sna_rows['data_point'] and concat the speedup_vs rows (NOT TESTED YET)
                        for sna_row in sna_rows['data_point']:
                            # Handle RVV calculations
                            if not rvv_rows.empty:
                                # Create a new copy for this iteration
                                rvv_rows_current = rvv_rows.copy()
                                rvv_rows_current['data_point'] = sna_row / rvv_rows['data_point']
                                rvv_rows_current = rvv_rows_current.assign(speedup_type='speedup_vs')
                                self.proc_data_speedup = pd.concat([self.proc_data_speedup, rvv_rows_current],
                                                                   ignore_index=True)
                            else:
                                print(f"Skipping N={unique_n} for {bid_hw_compiler} due to missing data rvv_rows.")

                        ## some benchmarks only have avx512 or avx2 data
                        # if rvv_rows.empty:
                        #    print(f"Skipping N={unique_n} for {bid_hw_compiler} due to missing data rvv_rows.")
                        #    continue
                        # else:
                        #    rvv_rows['data_point'] = sna_rows['data_point'].iloc[0] / rvv_rows['data_point']
                        #    rvv_rows = rvv_rows.assign(speedup_type='speedup_vs')
                        #    self.proc_data_speedup = pd.concat([self.proc_data_speedup, rvv_rows], ignore_index=True)

                # speedup_vv: rvv base with rvv uut (all auto-tuned)
                for bid_hw_compiler in unique_bid_hw_compiler:
                    for unique_n in unique_Ns:
                        # The benches have only Baselines (avx2 and avx512) and Ours (avx512). So we only have 1 speedup VV for avx512.
                        rvv_ours_rows = self.proc_data.loc[
                            (self.proc_data['benchId_hw_compiler'] == bid_hw_compiler) &
                            (self.proc_data['N'] == unique_n) &
                            (self.proc_data['run_type'] == 'best') &
                            (self.proc_data['name'].str.contains("RVV")) &
                            (self.proc_data['name'].str.contains("ours")) &
                            (self.proc_data['FLAG_AUTOTUNE_DISABLED'] == 0)
                            ]
                        rvv_baseline_rows = self.proc_data.loc[
                            (self.proc_data['benchId_hw_compiler'] == bid_hw_compiler) &
                            (self.proc_data['N'] == unique_n) &
                            (self.proc_data['run_type'] == 'best') &
                            (self.proc_data['name'].str.contains("RVV")) &
                            (self.proc_data['name'].str.contains("base")) &
                            (self.proc_data['FLAG_AUTOTUNE_DISABLED'] == 0)
                            ]
                        if rvv_ours_rows.empty or rvv_baseline_rows.empty:
                            print(
                                f"Skipping N={unique_n} for {bid_hw_compiler} due to missing data rvv_ours_rows or rvv_baseline_rows.")
                            continue
                        rvv_ours_rows.reset_index(drop=True, inplace=True)
                        rvv_baseline_rows.reset_index(drop=True, inplace=True)
                        # assert that the num of rows is the same
                        assert rvv_ours_rows.shape[0] == rvv_baseline_rows.shape[0]
                        rvv_ours_rows.loc[:, 'data_point'] = rvv_baseline_rows['data_point'] / rvv_ours_rows[
                            'data_point']  # speed up is ()^-1
                        rvv_ours_rows.loc[:, 'speedup_type'] = 'speedup_vv'
                        self.proc_data_speedup = pd.concat([self.proc_data_speedup, rvv_ours_rows],
                                                           ignore_index=True)

                # auto-tuning gain
                for bid_hw_compiler in unique_bid_hw_compiler:
                    for unique_n in unique_Ns:
                        # These benches have only Baselines (avx2 and rvv) and Ours (rvv). So we only have 1 gain for rvv.
                        rvv_ours_rows = self.proc_data.loc[
                            (self.proc_data['benchId_hw_compiler'] == bid_hw_compiler) &
                            (self.proc_data['N'] == unique_n) &
                            (self.proc_data['run_type'] == 'best') &
                            (self.proc_data['name'].str.contains("RVV")) &
                            (self.proc_data['name'].str.contains("ours")) &
                            (self.proc_data['FLAG_AUTOTUNE_DISABLED'] == 0)
                            ]
                        rvv_ours_no_autotune_rows = self.proc_data.loc[
                            (self.proc_data['benchId_hw_compiler'] == bid_hw_compiler) &
                            (self.proc_data['N'] == unique_n) &
                            (self.proc_data['run_type'] == 'best') &
                            (self.proc_data['name'].str.contains("RVV")) &
                            (self.proc_data['name'].str.contains("ours")) &
                            (self.proc_data['FLAG_AUTOTUNE_DISABLED'] == 1)
                            ]
                        if rvv_ours_rows.empty or rvv_ours_no_autotune_rows.empty:
                            print(
                                f"Skipping N={unique_n} for {bid_hw_compiler} due to missing data rvv_ours_rows or rvv_ours_no_autotune_rows.")
                            continue
                        rvv_ours_rows.reset_index(drop=True, inplace=True)
                        rvv_ours_no_autotune_rows.reset_index(drop=True, inplace=True)
                        # assert that the num of rows is the same
                        assert rvv_ours_rows.shape[0] == rvv_ours_no_autotune_rows.shape[0]
                        rvv_ours_rows.loc[:, 'data_point'] = rvv_ours_no_autotune_rows['data_point'] / \
                                                             rvv_ours_rows[
                                                                 'data_point']  # speed up is ()^-1
                        rvv_ours_rows.loc[:, 'speedup_type'] = 'autotuning_gain'
                        self.proc_data_speedup = pd.concat([self.proc_data_speedup, rvv_ours_rows],
                                                           ignore_index=True)


            elif bench_id == 4:
                print(f"NYI: Preprocessing data for benchID={bench_id}")
            else:
                print(f"Undefined benchID: {bench_id} for preprocessing.")

        self.proc_data_speedup['benchId_hw_compiler_name_speeduptype'] = \
            self.proc_data_speedup['benchId'].astype(str) + ';;' + \
            self.proc_data_speedup['hw'] + ';;' + \
            self.proc_data_speedup['compiler'] + ';;' + \
            translate_codename_to(self.proc_data_speedup['name']) + ';;' + \
            self.proc_data_speedup['speedup_type']

        self.proc_data_speedup['benchId_hw_compiler_speeduptype'] = \
            self.proc_data_speedup['benchId'].astype(str) + ';;' + \
            self.proc_data_speedup['hw'] + ';;' + \
            self.proc_data_speedup['compiler'] + ';;' + \
            self.proc_data_speedup['speedup_type']

    def plotgen_runtimes_all(self, reversed_text_order=False):
        """
        Generate all the plots, as many as needed for the parsed data.
        """
        self.preprocess_data()

        unique_n_list = self.proc_data['N'].unique()
        for n in unique_n_list:
            self.plotgen_runtimes_one(n, reversed_text_order)

    def plotgen_runtimes_one(self, n: int, reversed_text_order=False):
        """
        Generate a plot for a specific N.
        """
        # Extract the rows that have the specific N
        masked_data = self.proc_data[self.proc_data['N'] == n]

        # Extract the group names, each group is a unique (benchId, hw) pair
        group_names = masked_data['benchId_hw_compiler'].unique()
        group_names = sorted(group_names)

        # So we need to create k-bars and assign a group for each one.
        # First thing to do is to find how many bars we need to create.
        # Each bar has a unique (name, hw, benchId) combination. N is already fixed. So we use col `benchId_hw_compiler_name`.
        unique_bars = masked_data['benchId_hw_compiler_name'].unique()
        unique_bars = sorted(unique_bars)

        plt.figure(figsize=(12, 6))

        # reversed-text sorting
        if reversed_text_order:
            uniques = masked_data['benchId_hw_compiler_name'].unique()
            reversed_uniques = [x[::-1] for x in uniques]
            reversed_uniques.sort()
            order = [x[::-1] for x in reversed_uniques]
        else:
            order = sorted(masked_data['benchId_hw_compiler_name'].unique())

        barplot = sns.barplot(
            data=masked_data,
            x='benchId_hw_compiler_name',
            y='data_point',
            hue='benchId_hw_compiler',
            order=order,
            dodge=False,
            ci=95,  # Show 95% confidence intervals
            capsize=0.05  # Add caps to the error bars
        )

        # Add text on top of each bar
        for p in barplot.patches:
            if p.get_height() > 0:
                barplot.annotate(format(p.get_height(), '.3f'),
                                 (p.get_x() + p.get_width() / 2., p.get_height()),
                                 ha='left', va='center',
                                 xytext=(-3, 15),
                                 textcoords='offset points',
                                 rotation=90,
                                 fontsize=6)

        # Customize the plot
        plt.title(f"Runtimes for N={n}")
        plt.xlabel("Group")
        plt.xticks(rotation=90, fontsize=6)
        plt.ylabel("Runtime (ms)")
        lgd = plt.legend(title="Name", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(bottom=0.5, right=0.8)  # Adjust the bottom margin
        # plt.show()
        plt.savefig(f"{self.dir_out}/runtime_N_{n}_{reversed_text_order}.svg", bbox_extra_artists=(lgd,), bbox_inches='tight')

    def plotgen_speedups_all(self, reversed_text_order=False):
        """
        Generate all the plots, as many as needed for the parsed data.
        """
        self.preprocess_data()

        unique_n_list = self.proc_data_speedup['N'].unique()
        for n in unique_n_list:
            self.plotgen_speedups_one(n, reversed_text_order)

    def plotgen_speedups_one(self, n: int, reversed_text_order=False):
        # Extract the rows that have the specific N
        masked_data = self.proc_data_speedup[
            (self.proc_data_speedup['N'] == n)
        ]

        fig = plt.figure(figsize=(6, 6))

        # reversed-text sorting
        if reversed_text_order:
            uniques = masked_data['benchId_hw_compiler_name_speeduptype'].unique()
            reversed_uniques = [x[::-1] for x in uniques]
            reversed_uniques.sort()
            order = [x[::-1] for x in reversed_uniques]
        else:
            order = sorted(masked_data['benchId_hw_compiler_name_speeduptype'].unique())

        barplot = sns.barplot(
            data=masked_data,
            x='benchId_hw_compiler_name_speeduptype',
            y='data_point',
            hue='benchId_hw_compiler',
            palette='viridis',
            order=order,
            dodge=False,  # Do not set this to true. It will cause offset to the bars.
            ci=95,  # Show 95% confidence intervals
            capsize=0.05  # Add caps to the error bars
        )

        # Add text on top of each bar
        for p in barplot.patches:
            if p.get_height() > 0:
                barplot.annotate(format(p.get_height(), '.3f'),
                                 (p.get_x() + p.get_width() / 2., p.get_height()),
                                 ha='left', va='center',
                                 xytext=(-3, 15),
                                 textcoords='offset points',
                                 rotation=90,
                                 fontsize=6)

        # Customize the plot
        plt.title(f"Speedup for N={n}")
        plt.xlabel("Group")
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel("Speedup")
        lgd = plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(bottom=0.5, right=0.8)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{self.dir_out}/speedup_N_{n}_{reversed_text_order}.svg", bbox_extra_artists=(lgd,), bbox_inches='tight')

    def plotgen_speedups_over_N_all(self):
        """
        Generate the plot of speedups_vv (the most sensible vv case) over N for all the benchmarks, hw, and compilers.
        """

        self.preprocess_data()

        # Prepare the dataframe for the plot
        speedups_over_N = self.proc_data_speedup.copy()
        speedups_over_N = speedups_over_N[0:0]  # clear the rows

        for bench_id in self.proc_data_speedup['benchId'].unique():
            for hw in self.proc_data_speedup['hw'].unique():
                for compiler in self.proc_data_speedup['compiler'].unique():
                    if bench_id in [7, 8, 5, 6]:
                        # For these benchIds we only have 1 entry of speedup_vv, so we can just take any speed_vv entry.
                        masked_data = self.proc_data_speedup[
                            (self.proc_data_speedup['benchId'] == bench_id) &
                            (self.proc_data_speedup['hw'] == hw) &
                            (self.proc_data_speedup['compiler'] == compiler) &
                            (self.proc_data_speedup['speedup_type'] == 'speedup_vv')  # <----- any speed_vv entry
                            ]
                        speedups_over_N = pd.concat([speedups_over_N, masked_data], ignore_index=True)  # concat
                    else:
                        print(f"Skipping benchId={bench_id} for speedups_over_N.")
                        continue

        fig = plt.figure(figsize=(8, 6))
        lineplot = sns.lineplot(
            data=speedups_over_N,
            x='N',
            y='data_point',
            hue='benchId_hw_compiler',
            palette='viridis',
            ci=95,  # Show 95% confidence intervals
            markers=True,
            dashes=False
        )
        plt.title("Speedup_vv Over N")
        plt.xlabel("N")
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel("Speedup_vv")
        lgd = plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(bottom=0.5, right=0.8)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{self.dir_out}/speedup_vv_over_N.svg", bbox_extra_artists=(lgd,), bbox_inches='tight')


if __name__ == '__main__':
    # accept multiple instances of --dumps arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps', type=str, required=True, action='append')
    args = parser.parse_args()
    dumps = args.dumps

    obj = PlotSpeedUps(dumps, '/tmp')
    obj.plotgen_runtimes_all(reversed_text_order=True)
    obj.plotgen_runtimes_all(reversed_text_order=False)
    obj.plotgen_speedups_all(reversed_text_order=True)
    obj.plotgen_speedups_all(reversed_text_order=False)
    obj.plotgen_speedups_over_N_all()
