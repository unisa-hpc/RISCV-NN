import argparse
import inspect
import pathlib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.core.pylabtools import figsize
from click import style
from matplotlib.lines import Line2D
from parsing.parse import DumpsParser
from parsing.codebook import *
from parsing.lamda_funcs import *
import matplotlib as mpl

FORMAT = 'svg'
FIG_WIDTH = 8.27  # inches, A4 width=8.27
FIG_HEIGHT1 = 3.5
FIG_HEIGHT2 = 3.0


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
        self.is_preprocessed = False
        self.STYLE_BENCHID = "brief1"

    def serialize(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def deserialize(filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            print(f"Deserialized object of type {type(obj)}")
            return obj

    def load_data(self):
        self.dumps_parser.parse_all()
        self.raw_data = self.dumps_parser.get_dataframe_merged()
        self.raw_data['compiler'] = translate_compiler_name_to(self.raw_data['compiler'])
        self.raw_data['benchId'] = translate_benchId_to(self.raw_data['benchId'], self.STYLE_BENCHID)
        self.proc_data = self.raw_data.copy()

    def preprocess_data(self, export_to_excel=False):
        """
        Preprocess the raw data (self.raw_data) to get the speedups.

        {
            "Group": ["group_1"] * 3 + ["group_2"] * 4 + ["group_3"] * 2,
            "Bar Name": ["bar_1", "bar_2", "bar_3", "bar_4", "bar_5", "bar_6", "bar_7", "bar_8", "bar_9"],
            "Value": [5, 8, 3, 10, 7, 6, 2, 11, 9],  # Modified unique values for each bar
        }
        """
        if self.is_preprocessed:
            print("Data is already preprocessed.")
            return
        if self.raw_data is None:
            self.load_data()
        self._preprocess_add_columns()

        if export_to_excel:
            self.raw_data.to_excel(f"{self.dir_out}/raw_data.xlsx", index=False)
            self.proc_data.to_excel(f"{self.dir_out}/proc_data.xlsx", index=False)
            self.proc_data_speedup.to_excel(f"{self.dir_out}/proc_data_speedup.xlsx", index=False)

        self.is_preprocessed = True

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
            self.proc_data['name']

        self.proc_data['benchId_hw_compiler'] = \
            self.proc_data['benchId'].astype(str) + ', ' + \
            self.proc_data['hw'] + ', ' + \
            self.proc_data['compiler']

        self.proc_data['benchId_hw'] = \
            self.proc_data['benchId'].astype(str) + ', ' + \
            self.proc_data['hw']

        """
        The problem with speedups is that we CANNOT add them as new columns. We can have a speedup_type column.
        These will be added as rows to the speedup_type column:
        - Speedup_vv: Vectorized / Vectorized: avx2 base with avx2 uut, avx512 base with avx512 uut
        - Speedup_vs: Vectorized / Scalar: scalar no autovec base with avx2 uut, scalar no autovec base with avx512 uut
        - Speedup_ss: Scalar / Scalar: scalar no autovec base with scalar autovec base
        """
        unique_bids = [translate_str_benchId_to(e, self.STYLE_BENCHID, reverse=True) for e in
                       self.proc_data['benchId'].unique()]
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
            self.proc_data_speedup['name'] + ';;' + \
            self.proc_data_speedup['speedup_type']

        self.proc_data_speedup['benchId_hw_compiler_speeduptype'] = \
            self.proc_data_speedup['benchId'].astype(str) + ';;' + \
            self.proc_data_speedup['hw'] + ';;' + \
            self.proc_data_speedup['compiler'] + ';;' + \
            self.proc_data_speedup['speedup_type']

        self.proc_data_speedup['benchId_speeduptype'] = \
            self.proc_data_speedup['benchId'].astype(str) + ', ' + \
            self.proc_data_speedup['speedup_type']

    def plotgen_runtimes_all(self, reversed_text_order=False, per_hw=False):
        """
        Generate all the plots, as many as needed for the parsed data.
        """
        self.preprocess_data()
        if not per_hw:
            unique_n_list = self.proc_data['N'].unique()
            for n in unique_n_list:
                self.plotgen_runtimes_one(n, reversed_text_order)
        else:
            unique_hw_list = self.proc_data['hw'].unique()
            for hw in unique_hw_list:
                unique_n_list = self.proc_data['N'].unique()
                for n in unique_n_list:
                    self.plotgen_runtimes_one(n, reversed_text_order, hw)

    def plotgen_runtimes_one(self, n: int, reversed_text_order=False, hw: str = None):
        """
        Generate a plot for a specific N.
        """
        # Extract the rows that have the specific N
        if hw is None:
            cond = self.proc_data['N'] == n
        else:
            cond = (self.proc_data['N'] == n) & (self.proc_data['hw'] == hw)

        masked_data = self.proc_data[cond]

        # Extract the group names, each group is a unique (benchId, hw) pair
        group_names = masked_data['benchId_hw_compiler'].unique()
        group_names = sorted(group_names)

        # So we need to create k-bars and assign a group for each one.
        # First thing to do is to find how many bars we need to create.
        # Each bar has a unique (name, hw, benchId) combination. N is already fixed. So we use col `benchId_hw_compiler_name`.
        unique_bars = masked_data['benchId_hw_compiler_name'].unique()
        unique_bars = sorted(unique_bars)

        plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT1))

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
            ci="sd",  # Show std-deviation confidence intervals
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
        plt.savefig(f"{self.dir_out}/runtime_N_{n}_{reversed_text_order}_{hw}.{FORMAT}", bbox_extra_artists=(lgd,),
                    bbox_inches='tight', dpi=300)

    def plotgen_speedups_all(self, reversed_text_order=False, per_hw=False):
        """
        Generate all the plots, as many as needed for the parsed data.
        """
        self.preprocess_data()

        if not per_hw:
            unique_n_list = self.proc_data_speedup['N'].unique()
            for n in unique_n_list:
                self.plotgen_speedups_one(n, reversed_text_order)
        else:
            unique_hw_list = self.proc_data_speedup['hw'].unique()
            for hw in unique_hw_list:
                unique_n_list = self.proc_data_speedup['N'].unique()
                for n in unique_n_list:
                    self.plotgen_speedups_one(n, reversed_text_order, hw=hw)

    def plotgen_speedups_type2_all(self, reversed_text_order=False, hw: [str] = None):
        """
        Generate all the speedup plots (type2)
        """
        self.preprocess_data()

        unique_n_list = self.proc_data_speedup['N'].unique()
        unique_compiler_list = self.proc_data_speedup['compiler'].unique()
        for n in unique_n_list:
            for c in unique_compiler_list:
                self.plotgen_speedups_type2_one_multi(n, c, reversed_text_order, hw=hw)

    def plotgen_speedups_type2_sidebysidecompilers_for_all_N(
            self,
            reversed_text_order=False,
            hw: [str] = None,
            compilers=[str],
            y_range=None
    ):
        """
        Generate all the speedup plots (type2)
        """
        self.preprocess_data()

        assert len(compilers) == 2, "Only two compilers are supported for this plot."

        unique_n_list = self.proc_data_speedup['N'].unique()
        for n in unique_n_list:
            if len(hw)>1:
                self.plotgen_speedups_type2_one_multi(n, compilers[0], reversed_text_order, hw=hw, y_axis_range=y_range, hide_legend=True)
                self.plotgen_speedups_type2_one_multi(n, compilers[1], reversed_text_order, hw=hw, y_axis_range=y_range, hide_legend=False)
            else:
                self.plotgen_speedups_type2_one_single(n, compilers[0], reversed_text_order, hw=hw, y_axis_range=y_range, hide_legend=True)
                self.plotgen_speedups_type2_one_single(n, compilers[1], reversed_text_order, hw=hw, y_axis_range=y_range, hide_legend=False)

    def plotgen_speedups_all_per_n_subplots(self, reversed_text_order=False, per_hw=False):
        """
        Generate all the plots, as many as needed for the parsed data.
        per_hw: If True, generate a separate plot for each hardware.
        """
        self.preprocess_data()

        unique_hw_list = self.proc_data_speedup['hw'].unique() if per_hw else [None]

        for hw in unique_hw_list:
            if per_hw:
                sub = self.proc_data_speedup[self.proc_data_speedup['hw'] == hw]
                unique_n_list = sub['N'].unique()
            else:
                unique_n_list = self.proc_data_speedup['N'].unique()

            # Sort the unique Ns
            unique_n_list.sort()

            # create a figure with multiple subplots
            fig, axs = plt.subplots(len(unique_n_list), 1, figsize=(FIG_WIDTH, 32))
            fig.subplots_adjust(hspace=0.5, wspace=0.8)

            if len(unique_n_list) == 1:
                axs = [axs]

            x_ticks_list = []

            for i, n in enumerate(unique_n_list):
                ax, lgnd = self.plotgen_speedups_one(n, reversed_text_order, ax=axs[i], hw=hw)
                x_tick_texts = [tick.get_text() for tick in ax.get_xticklabels()]
                x_ticks_list.append(x_tick_texts)

            if all(x_ticks == x_ticks_list[0] for x_ticks in x_ticks_list):
                # If all x-tick labels are the same, remove them from all but the last subplot
                for i, ax in enumerate(axs[:-1]):
                    ax.set_xticklabels([])  # Remove x-tick labels

            # each subplot has its own legend
            # save the figure with all the legends and subplots
            plt.savefig(f"{self.dir_out}/speedup_all_N_{hw}_{reversed_text_order}.{FORMAT}", bbox_extra_artists=(lgnd,),
                        bbox_inches='tight', dpi=300)

    def plotgen_speedups_one(self, n: int, reversed_text_order=False, ax=None, hw: str = None):
        if hw is None:
            cond = self.proc_data_speedup['N'] == n
        else:
            cond = (self.proc_data_speedup['N'] == n) & (self.proc_data_speedup['hw'] == hw)

        # Extract the rows that have the specific N
        masked_data = self.proc_data_speedup[cond]

        save_fig = False  # Track whether to save the figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT1))
            fig.subplots_adjust(bottom=0.5, right=0.8)
            save_fig = True

        # reversed-text sorting
        if reversed_text_order:
            uniques = masked_data['benchId_hw_compiler_name_speeduptype'].unique()
            reversed_uniques = [x[::-1] for x in uniques]
            reversed_uniques.sort()
            order = [x[::-1] for x in reversed_uniques]
        else:
            order = sorted(masked_data['benchId_hw_compiler_name_speeduptype'].unique())

        barplot = sns.barplot(
            ax=ax,
            data=masked_data,
            x='benchId_hw_compiler_name_speeduptype',
            y='data_point',
            hue='benchId_hw_compiler',
            palette='viridis',
            order=order,
            dodge=False,  # Do not set this to true. It will cause offset to the bars.
            ci="sd",  # Show std-deviation confidence intervals
            capsize=0.05  # Add caps to the error bars
        )

        # Add text on top of each bar
        for p in barplot.patches:
            if p.get_height() > 0:
                ax.annotate(format(p.get_height(), '.3f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='left', va='center',
                            xytext=(-3, 15),
                            textcoords='offset points',
                            rotation=90,
                            fontsize=6)

        # Customize the plot
        ax.set_title(f"Speedup for N={n}")
        ax.set_xlabel("Group")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
        ax.set_ylabel("Speedup")
        lgd = ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save figure only if we created it
        if save_fig:
            save_path = f"{self.dir_out}/speedup_N_{n}_{reversed_text_order}.{FORMAT}"
            fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
            plt.close(fig)  # Avoid memory leak

        return ax, lgd  # Return ax and legend for further customization

    def plotgen_speedups_type2_one_multi(
            self,
            fixed_N: int,
            fixed_compiler: str,
            reversed_text_order=False,
            ax=None,
            hw: [str] = None,
            y_axis_range: tuple[int, int] = None,
            hide_legend=False
    ):
        if hw is None:
            cond = (self.proc_data_speedup['N'] == fixed_N) & (self.proc_data_speedup['compiler'] == fixed_compiler)
        else:
            cond = (self.proc_data_speedup['N'] == fixed_N) & (self.proc_data_speedup['compiler'] == fixed_compiler) & (
                self.proc_data_speedup['hw'].isin(hw))

        # Sort hw
        #hw = sorted(hw)

        # Extract the filtered data
        masked_data = self.proc_data_speedup[cond]

        # Delete autotuning_gain column
        masked_data = masked_data[masked_data['speedup_type'] != 'autotuning_gain']

        save_fig = False  # Track whether to save the figure
        if ax is None:
            save_fig = True

        # Sorting benchmark names
        unique_benchmarks = masked_data['benchId_speeduptype'].unique()
        if reversed_text_order:
            order = sorted(unique_benchmarks, key=lambda x: x[::-1])  # Sort by reversed string
        else:
            order = sorted(unique_benchmarks)

        # Skip if masked_data is empty
        if masked_data.empty:
            print(f"Skipping N={fixed_N} for compiler={fixed_compiler} due to empty masked_data.")
            return None, None

        # Create the seaborn catplot (FacetGrid)
        catplot = sns.catplot(
            data=masked_data,
            order=order,
            col_order=hw,
            kind="bar",
            x="benchId_speeduptype",
            y="data_point",
            col="hw",
            hue="speedup_type",
            height=FIG_HEIGHT1,
            aspect=1,
        )

        # Set y-axis range if provided
        if y_axis_range is not None:
            catplot.set(ylim=y_axis_range)

        # Adjust the legend position outside the plot
        catplot._legend.set_bbox_to_anchor((0,0,1.4,1.4)) # (left, bottom, right, top)
        catplot._legend.set_loc("upper right")
        catplot._legend.set

        catplot._legend.set_title("Speedup Type")
        catplot._legend.set_frame_on(True)  # Enable legend box


        # Rotate x-ticks labels for each subplot
        for ax in catplot.axes.flat:
            ax.set_xlabel("")  # Remove x-axis label
            x_lbls = ax.get_xticklabels()
            # change the strings of the Text objects
            x_lbls_short = [x.get_text().split(',')[0].replace('FPoT', '') for x in x_lbls]
            ax.set_xticklabels(x_lbls_short, rotation=90, fontsize=9)

        for ax in catplot.axes.flat:  # Loop through all subplots
            for p in ax.patches:  # Loop through each bar
                height = p.get_height()
                if height > 0:
                    ax.annotate(f'{height:.3f}',  # Format to 3 decimal places
                                (p.get_x() + p.get_width() / 2., height),  # Position at the top of the bar
                                ha='center', va='bottom',  # Center align
                                xytext=(0, FIG_HEIGHT1),  # Offset a bit above the bar
                                textcoords='offset points',
                                fontsize=9, rotation=90)  # Rotate 90 degrees

        # Set title for the entire plot
        catplot.fig.set_size_inches(FIG_WIDTH/2, FIG_HEIGHT1)
        catplot.fig.suptitle(f"Speedup for N={fixed_N} and Compiler={fixed_compiler}", fontsize=10, y=1.15)

        # Move "hw=xxxx" titles higher
        catplot.set_titles("{col_name}", size=9, y=1.10)
        # Adjust layout using tight_layout to leave space for suptitle
        catplot.fig.tight_layout(rect=[0, 0, 1.5, 1.2]) # (left, bottom, right, top) # does not affect the legend pos
        # set y-label
        catplot.set_ylabels("Speedup")

        if hide_legend:
            catplot._legend.remove()
        else:
            # also remove the y-axis label
            for ax in catplot.axes.flat:
                ax.set_ylabel("")

        # Save figure if it was created inside this function
        if save_fig:
            if y_axis_range is None:
                save_path = f"{self.dir_out}/speedup_type2_N={fixed_N}_compiler={fixed_compiler}_hw{hw}_reverse{reversed_text_order}.{FORMAT}"
            else:
                save_path = f"{self.dir_out}/speedup_type2_N={fixed_N}_compiler={fixed_compiler}_hw{hw}_yrange_{y_axis_range}_legend{hide_legend}_reverse{reversed_text_order}.{FORMAT}"
            catplot.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(catplot.fig)  # Avoid memory leaks

        return catplot, catplot._legend  # Return catplot object and legend for further customization

    def plotgen_speedups_type2_one_single(
            self,
            fixed_N: int,
            fixed_compiler: str,
            reversed_text_order=False,
            ax=None,
            hw: [str] = None,
            y_axis_range: tuple[int, int] = None,
            hide_legend=False
    ):
        if hw is None:
            cond = (self.proc_data_speedup['N'] == fixed_N) & (self.proc_data_speedup['compiler'] == fixed_compiler)
        else:
            cond = (self.proc_data_speedup['N'] == fixed_N) & (self.proc_data_speedup['compiler'] == fixed_compiler) & (
                self.proc_data_speedup['hw'].isin(hw))

        # Sort hw
        hw = sorted(hw)

        # Extract the filtered data
        masked_data = self.proc_data_speedup[cond]

        # Delete autotuning_gain column
        masked_data = masked_data[masked_data['speedup_type'] != 'autotuning_gain']

        save_fig = False  # Track whether to save the figure
        if ax is None:
            save_fig = True

        # Sorting benchmark names
        unique_benchmarks = masked_data['benchId_speeduptype'].unique()
        if reversed_text_order:
            order = sorted(unique_benchmarks, key=lambda x: x[::-1])  # Sort by reversed string
        else:
            order = sorted(unique_benchmarks)

        # Skip if masked_data is empty
        if masked_data.empty:
            print(f"Skipping N={fixed_N} for compiler={fixed_compiler} due to empty masked_data.")
            return None, None

        # Create the seaborn catplot (FacetGrid)
        catplot = sns.catplot(
            data=masked_data,
            order=order,
            col_order=hw,
            kind="bar",
            x="benchId_speeduptype",
            y="data_point",
            col="hw",
            hue="speedup_type",
            height=FIG_HEIGHT1/3 * 2,
            aspect=1,
        )

        # Set y-axis range if provided
        if y_axis_range is not None:
            catplot.set(ylim=y_axis_range)

        # Adjust the legend position outside the plot
        catplot._legend.set_bbox_to_anchor((0,0,1.3,0.9)) # (left, bottom, right, top)
        catplot._legend.set_loc("upper right")
        catplot._legend.set

        catplot._legend.set_title("Speedup Type")
        catplot._legend.set_frame_on(True)  # Enable legend box


        # Rotate x-ticks labels for each subplot
        for ax in catplot.axes.flat:
            ax.set_xlabel("")  # Remove x-axis label
            x_lbls = ax.get_xticklabels()
            # change the strings of the Text objects
            x_lbls_short = [x.get_text().split(',')[0].replace('FPoT', '') for x in x_lbls]
            ax.set_xticklabels(x_lbls_short, rotation=90, fontsize=9)

        for ax in catplot.axes.flat:  # Loop through all subplots
            for p in ax.patches:  # Loop through each bar
                height = p.get_height()
                if height > 0:
                    ax.annotate(f'{height:.3f}',  # Format to 3 decimal places
                                (p.get_x() + p.get_width() / 2., height),  # Position at the top of the bar
                                ha='center', va='bottom',  # Center align
                                xytext=(0, FIG_HEIGHT1),  # Offset a bit above the bar
                                textcoords='offset points',
                                fontsize=9, rotation=90)  # Rotate 90 degrees

        # Set title for the entire plot
        catplot.fig.set_size_inches(FIG_WIDTH/4, FIG_HEIGHT1)
        catplot.fig.suptitle(f"Speedup for N={fixed_N} and\nCompiler={fixed_compiler}", fontsize=10, x=0.25 , y=1.15, ha='left')

        # Move "hw=xxxx" titles higher
        catplot.set_titles("{col_name}", size=9, y=1.10)
        # Adjust layout using tight_layout to leave space for suptitle
        catplot.fig.tight_layout(rect=[0, 0, 1.5, 1.2]) # (left, bottom, right, top) # does not affect the legend pos
        # set y-label
        catplot.set_ylabels("Speedup")

        if hide_legend:
            catplot._legend.remove()
        else:
            # also remove the y-axis label
            for ax in catplot.axes.flat:
                ax.set_ylabel("")

        # Save figure if it was created inside this function
        if save_fig:
            if y_axis_range is None:
                save_path = f"{self.dir_out}/speedup_type2_N={fixed_N}_compiler={fixed_compiler}_hw{hw}_reverse{reversed_text_order}.{FORMAT}"
            else:
                save_path = f"{self.dir_out}/speedup_type2_N={fixed_N}_compiler={fixed_compiler}_hw{hw}_yrange_{y_axis_range}_legend{hide_legend}_reverse{reversed_text_order}.{FORMAT}"
            catplot.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(catplot.fig)  # Avoid memory leaks

        return catplot, catplot._legend  # Return catplot object and legend for further customization

    def plotgen_speedups_over_N_all_multilines_in_single_plot(self, hw_groups=[]):
        """
        Generate the plot of speedups_vv (the most sensible vv case) over N for all the benchmarks, hw, and compilers.
        hw_groups is a list of lists, where each list contains the hw names that should be grouped together.
        """
        self.preprocess_data()

        if len(hw_groups) == 0:
            hw_groups.append(self.proc_data_speedup['hw'].unique())

        for hw_group in hw_groups:

            # Start with a fresh copy
            speedups_over_N = self.proc_data_speedup.copy()
            speedups_over_N = speedups_over_N[0:0]  # clear the rows

            for bench_id in [translate_str_benchId_to(e, self.STYLE_BENCHID, reverse=True) for e in
                             self.proc_data_speedup['benchId'].unique()]:
                for hw in hw_group:
                    for compiler in self.proc_data_speedup['compiler'].unique():
                        if bench_id in [7, 8, 5, 6]:
                            # For these benchIds we only have 1 entry of speedup_vv, so we can just take any speed_vv entry.
                            masked_data = self.proc_data_speedup[
                                (self.proc_data_speedup['benchId'] == translate_str_benchId_to(bench_id,
                                                                                               self.STYLE_BENCHID)) &
                                (self.proc_data_speedup['hw'] == hw) &
                                (self.proc_data_speedup['compiler'] == compiler) &
                                (self.proc_data_speedup['speedup_type'] == 'speedup_vv')  # <----- any speed_vv entry
                                ]
                            speedups_over_N = pd.concat([speedups_over_N, masked_data], ignore_index=True)  # concat
                        else:
                            print(f"Skipping benchId={bench_id} for speedups_over_N.")
                            continue

            fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT1))
            lineplot = sns.lineplot(
                data=speedups_over_N,
                x='N',
                y='data_point',
                hue='benchId_hw',
                style='compiler',
                palette='viridis',
                ci="sd",  # Show std-deviation confidence intervals
                markers=False,
                dashes=True,
                legend='full'
            )
            lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.title("Speedup_vv Over N")
            plt.xlabel("N")
            plt.xticks(rotation=90, fontsize=7)
            plt.ylabel("Speedup_vv")
            plt.subplots_adjust(bottom=0.5, right=0.8)
            plt.tight_layout()
            # plt.show()
            plt.savefig(f"{self.dir_out}/speedup_vv_over_N__{str(hw_group)}.{FORMAT}", bbox_extra_artists=(lgd,),
                        bbox_inches='tight', dpi=300)

    def plotgen_speedups_over_N_all_as_subfigures(self, hw_list=[], inset_zoom=False):
        """
        Generate the plot of speedups_vv (the most sensible vv case) over N for all the benchmarks, hw, and compilers.
        hw_groups is a list of lists, where each list contains the hw names that should be grouped together.
        """

        self.preprocess_data()

        if len(hw_list) == 0:
            hw_list.append(self.proc_data_speedup['hw'].unique())

        # Create a figure with multiple subplots
        fig, axs = plt.subplots(len(hw_list), 1, figsize=(FIG_WIDTH, len(hw_list) * FIG_HEIGHT2))
        fig.subplots_adjust(hspace=0.15)

        if len(hw_list) == 1:
            axs = [axs]

        for i, hw in enumerate(hw_list):
            masked_data = self.proc_data_speedup[
                (self.proc_data_speedup['hw'] == hw) &
                (self.proc_data_speedup['speedup_type'] == 'speedup_vv')  # <----- any speed_vv entry
                ]
            lineplot = sns.lineplot(
                ax=axs[i],
                data=masked_data,
                x='N',
                y='data_point',
                hue='benchId_hw',
                style='compiler',
                palette='viridis',
                ci="sd",  # Show std-deviation confidence intervals
                markers=False,
                dashes=True,
                legend='full'
            )

            if inset_zoom:
                ############################################################################
                # Add  a zoomed-in inset to top-left corner
                # If there is only one subfigure, we dont need to add a zoomed-in inset
                if len(hw_list) != 1:
                    zommed_data = masked_data[(masked_data['N'] >= 128) & (masked_data['N'] <= 512)]
                    axins = axs[i].inset_axes([0.05, .6, 0.3, 0.4])
                    # axins.set_yscale('log')
                    sns.lineplot(
                        ax=axins,
                        # 64<=N<=512
                        data=zommed_data,
                        x='N',
                        y='data_point',
                        hue='benchId_hw',
                        style='compiler',
                        palette='viridis',
                        ci="sd",  # Show std-deviation confidence intervals
                        markers=False,
                        dashes=True,
                        legend=False,
                    )

                    axins.set_xticks(zommed_data['N'].unique())
                    axins.set_xticklabels(zommed_data['N'].unique(), rotation=90, fontsize=6)

                    # set y tick labels font size only
                    axins.tick_params(axis='y', labelsize=6)

                    axins.grid(True)
                    axins.set_xlabel("")
                    axins.set_ylabel("")

                    # set color for border and ticks
                    for spine in axins.spines.values():
                        spine.set_edgecolor('blue')
                    axins.tick_params(axis='x', colors='blue')
                    axins.tick_params(axis='y', colors='blue')
                ############################################################################

            # if i==0: axs[i].set_title("Speedup_vv Over N")
            if i == len(hw_list) - 1:
                axs[i].set_xlabel("Square Matrix Size (N)")
            else:
                axs[i].set_xlabel("")

            axs[i].set_xticks(masked_data['N'].unique())
            if i != len(hw_list) - 1:
                # hide x-tick labels only, keep grid
                axs[i].set_xticklabels([])
            axs[i].tick_params(axis='x', rotation=90, labelsize=6)
            axs[i].set_ylabel("Speedup")  # Explain in the caption of the figure that this is speedup_vv
            axs[i].grid(True)

            # Modify the legend manually without touching the dataframe
            new_labels = {'benchId_hw': 'Kernel:', 'compiler': 'Compiler:'}

            def omit_hw(s):
                return s.split(',')[0]

            handles, labels = axs[i].get_legend_handles_labels()
            lgd = axs[i].legend(handles,
                                [omit_hw(label) if label not in new_labels else new_labels[label] for label in labels],
                                bbox_to_anchor=(1.05, 1), loc='upper left')

            # Add bold text to bottom right corner of each subplot to indicate the hardware
            # We dont want to put the HW name in the legend!
            axs[i].text(0.95, 0.05, hw, fontsize=8, fontweight='bold', transform=axs[i].transAxes, ha='right')

        fig.savefig(f"{self.dir_out}/speedup_vv_over_N_subfig__{str(hw_list)}.{FORMAT}", bbox_extra_artists=(lgd,),
                    bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    # accept multiple instances of --dumps arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--dumps', type=str, required=False, action='append')
    parser.add_argument('--s-from', type=str, required=False, help='Restore the class state from a pickle file.')
    parser.add_argument('--s-to', type=str, required=False, help='Save the class state to a pickle file.')

    args = parser.parse_args()
    sns.set_theme(font_scale=1)

    # dumps and s-from are mutually exclusive
    if args.dumps is None and args.s_from is None:
        parser.error('At least one of --dumps or --s-from is required.')
    if args.dumps is not None and args.s_from is not None:
        # check if s-from file exists, if it does, ignore dumps, else ignore s-from, use pathlib
        if not pathlib.Path(args.s_from).is_file():
            print(f"File {args.s_from} does not exist.")
            print("Ignoring --s-from.")
            args.s_from = None
        else:
            print("When --s-from is provided, --dumps will be ignored.")
            args.dumps = None

    if args.s_from is not None:
        obj = PlotSpeedUps.deserialize(args.s_from)
    else:
        dumps = args.dumps
        obj = PlotSpeedUps(dumps, '/tmp')

    """
    # Temp. disabled to speedup debugging.
    
    for order in [True, False]:
        for per_hw in [True, False]:
            # These work, but we no longer need them
            #obj.plotgen_runtimes_all(reversed_text_order=order, per_hw=per_hw)
            #obj.plotgen_speedups_all(reversed_text_order=order, per_hw=per_hw)
            #obj.plotgen_speedups_all_per_n_subplots(reversed_text_order=order, per_hw=per_hw)
        obj.plotgen_speedups_type2_all(reversed_text_order=order, hw=['Xeon5218', 'Xeon8260', 'Ryzen97950X']) # or hw=['cpu1', 'cpu2']
        obj.plotgen_speedups_type2_all(reversed_text_order=order, hw=['SpacemitK1']) # or hw=['cpu1', 'cpu2']
    """
    obj.plotgen_speedups_type2_all(
        reversed_text_order=False,
        hw=['Xeon5218', 'Xeon8260', 'Ryzen97950X']
    )

    obj.plotgen_speedups_type2_all(
        reversed_text_order=False,
        hw=['SpacemitK1']
    )

    obj.plotgen_speedups_type2_sidebysidecompilers_for_all_N(
        reversed_text_order=False,
        hw=['Xeon5218', 'Xeon8260', 'Ryzen97950X'],
        compilers=['LLVM18', 'GCC14.2'],
        y_range=(0, 12)
    )

    obj.plotgen_speedups_type2_sidebysidecompilers_for_all_N(
        reversed_text_order=False,
        hw=['SpacemitK1'],
        compilers=['LLVM18', 'GCC14.2'],
        y_range=(0, 12)
    )

    """
    # These work, but we dont need them anymore.
    
    obj.plotgen_speedups_over_N_all_multilines_in_single_plot()
    # 'Xeon5218' 'Xeon8260' 'Xeon8358' 'SpacemitK1'
    obj.plotgen_speedups_over_N_all_multilines_in_single_plot(
        [
            ['SpacemitK1'],             # SpacemitK1
            ['Ryzen97950X'],            # Pagamp
            ['Xeon8260'],               # G100
            ['Xeon5218'],               # Furore
            ['Xeon5218', 'Xeon8260'],   # Furore, G100
        ]
    )
    """

    obj.plotgen_speedups_over_N_all_as_subfigures(hw_list=
    [
        'Xeon5218',
        'Xeon8260',
        'Ryzen97950X'
    ]
    )
    obj.plotgen_speedups_over_N_all_as_subfigures(hw_list=
    [
        'SpacemitK1',
    ]
    )

    if args.s_to is not None:
        obj.serialize(args.s_to)
