#
# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0
#

#
# Created by saleh.
#

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import json

class PlotSpeedUpsGpuCuda:
    def __init__(self, path):
        self.path = path

        # Look for all json files in self.path
        self.files = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".json"):
                    self.files.append(os.path.join(root, file))
        self.files.sort()

        self.data_raw_keys = ["KernelMatmulBase", "KernelMatmulPotUint8Packed2", "KernelMatmulPotUint8Packed4"]
        self.data_raw = {}
        for name in self.data_raw_keys:
            found = False
            for file in self.files:
                if name in file:
                    # load json file into python dict
                    with open(file, 'r') as f:
                        d = json.load(f)
                    self.data_raw = self.merge_dicts(self.data_raw, d)
                    found = True
                    break
            if not found:
                raise ValueError(f"Could not find {name} in the files.")
        self.data_proc = {}
        self.is_preprocessed = False

    def merge_dicts(self, dict1, dict2):
        """Merges two dictionaries into one, handling nested dicts, lists, and other iterables."""
        merged = dict1.copy()  # Create a copy of dict1 to avoid modifying it
        for key, value in dict2.items():
            if key in merged:
                if isinstance(merged[key], dict) and isinstance(value, dict):
                    # If both values are dictionaries, recursively merge them
                    merged[key] = self.merge_dicts(merged[key], value)
                elif isinstance(merged[key], list) and isinstance(value, list):
                    # If both values are lists, concatenate them
                    merged[key] = merged[key] + value
                else:
                    # Otherwise, simply overwrite the value in dict1 with the value from dict2
                    merged[key] = value
            else:
                # If the key doesn't exist in dict1, just add it
                merged[key] = value
        return merged

    def preprocess_data(self):
        if self.is_preprocessed:
            print("Data is already preprocessed.")
            return

        self.data_proc = self.data_raw["kernels.cu"]

        # the first level keys ---> split by `<` and keep [0]
        self.data_proc = {k.split("<")[0]: v for k, v in self.data_proc.items()}

        self.is_preprocessed = True

    def plotgen_speedups_all(self):
        self.preprocess_data()
        base = self.data_proc["KernelMatmulBase"]
        pot_pack2 = self.data_proc["KernelMatmulPotUint8Packed2"]
        pot_pack4 = self.data_proc["KernelMatmulPotUint8Packed4"]

        # be sure we have matching 1st level keys
        assert base.keys() == pot_pack2.keys() == pot_pack4.keys()

        df = pd.DataFrame(columns=["kernel", "n", "time"])

        for n in base.keys():
            base_times = base[n]["best"]["times"]
            pot_pack2_times = pot_pack2[n]["best"]["times"]
            pot_pack4_times = pot_pack4[n]["best"]["times"]

            for t in base_times:
                df.loc[len(df)] = ["base", n, t]
            for t in pot_pack2_times:
                df.loc[len(df)] = ["pot_pack2", n, t]
            for t in pot_pack4_times:
                df.loc[len(df)] = ["pot_pack4", n, t]

        df_speedups = pd.DataFrame(columns=["speedup_type", "n", "speedup"])
        for n in base.keys():
            base_times = base[n]["best"]["times"]
            pot_pack2_times = pot_pack2[n]["best"]["times"]
            pot_pack4_times = pot_pack4[n]["best"]["times"]

            # make sure we have the same number of times for all three kernels
            assert len(base_times) == len(pot_pack2_times) == len(pot_pack4_times)

            # speedup of pot_pack2_times over base_times
            # for two N elements of times, compute N^2 speedups
            # So for each time in pot_pack2, compute N speedups for base
            for i in range(len(pot_pack2_times)):
                for j in range(len(base_times)):
                    df_speedups.loc[len(df_speedups)] = ["pot_pack2_over_base", n, base_times[j] / pot_pack2_times[i]]

            # speedup of pot_pack4_times over base_times
            for i in range(len(pot_pack4_times)):
                for j in range(len(base_times)):
                    df_speedups.loc[len(df_speedups)] = ["pot_pack4_over_base", n, base_times[j] / pot_pack4_times[i]]

        # bar graph of speedups for different N and different types of speedups in one plot
        sns.set_theme(style="whitegrid")
        sns.set_context("paper")
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x="n", y="speedup", hue="speedup_type", data=df_speedups)
        ax.set_title("Speedups of different kernels over base kernel")
        plt.show()




if __name__ == '__main__':
    # This script takes the path to the sub-dump dir of the auto-tuner script for gpu_cuda.
    # Only one arg, cannot feed multiple sub-dumps.
    parser = argparse.ArgumentParser()
    parser.add_argument('--subdump', type=str, required=True)
    args = parser.parse_args()

    plot = PlotSpeedUpsGpuCuda(args.subdump)
    plot.plotgen_speedups_all()