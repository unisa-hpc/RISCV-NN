import orjson
import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib


class GpuAutotunerConvergencePlotter:
    def __init__(self, subdumpdir):
        self.subdumpdir = subdumpdir
        self.file_paths = self.get_all_json_files()
        self.data_raw = {}

        # file -> kernel -> N -> {
        #                   'best': [..., times:[...], time: 000],
        #                   'env',
        #                   'all': [..., times:[...], time: 000],
        #                   'opt': {'opt: str, 'maxiter': int, 'time_limit': int}
        #                   }
        self.data_aggr = {}

        self.load_data()
        self.aggregate_raw_data()
        print(self.data_raw.keys().__str__())

    def load_data(self):
        # open json files and parse them with orjson with try except
        for file in self.file_paths:
            with open(file, 'r') as f:
                try:
                    data = orjson.loads(f.read())
                    self.data_raw[pathlib.Path(file).name] = data
                except Exception as e:
                    print(f"Error reading file {file}: {e}")

    def aggregate_raw_data(self):
        for file, data in self.data_raw.items():
            for kernel_file, kernels in data.items():  # `kernels` contains multiple kernel names
                if kernel_file not in self.data_aggr:
                    self.data_aggr[kernel_file] = {}

                for kernel_name, kernel_data in kernels.items():  # Iterate through kernel names
                    if kernel_name not in self.data_aggr[kernel_file]:
                        self.data_aggr[kernel_file][kernel_name] = {}

                    for N, results in kernel_data.items():  # Iterate through N values
                        if N not in self.data_aggr[kernel_file][kernel_name]:
                            self.data_aggr[kernel_file][kernel_name][N] = {
                                'best': [],
                                'env': None,
                                'all': [],
                                'opt': None
                            }

                        # Safely get 'best', 'all', and 'env' values
                        best_results = results.get('best', [])  # Default to empty list if missing
                        all_results = results.get('all', [])  # Default to empty list if missing
                        env_value = results.get('env', None)  # Default to None if missing
                        opt_value = results.get('opt', None)  # Default to None if missing

                        # Ensure 'best' and 'all' are lists before extending
                        if isinstance(best_results, list):
                            self.data_aggr[kernel_file][kernel_name][N]['best'].extend(best_results)
                        else:
                            self.data_aggr[kernel_file][kernel_name][N]['best'].append(best_results)

                        if isinstance(all_results, list):
                            self.data_aggr[kernel_file][kernel_name][N]['all'].extend(all_results)
                        else:
                            self.data_aggr[kernel_file][kernel_name][N]['all'].append(all_results)

                        # Handle 'env' - store only if consistent, otherwise raise a warning
                        if self.data_aggr[kernel_file][kernel_name][N]['env'] is None:
                            self.data_aggr[kernel_file][kernel_name][N]['env'] = env_value
                        elif self.data_aggr[kernel_file][kernel_name][N]['env'] != env_value and env_value is not None:
                            print(f"Warning: Inconsistent 'env' values for kernel {kernel_file}/{kernel_name}, N={N}")

                        # Handle 'opt' - store only if consistent, otherwise raise a warning
                        if self.data_aggr[kernel_file][kernel_name][N]['opt'] is None:
                            self.data_aggr[kernel_file][kernel_name][N]['opt'] = opt_value
                        elif self.data_aggr[kernel_file][kernel_name][N]['opt'] != opt_value and opt_value is not None:
                            print(f"Warning: Inconsistent 'opt' values for kernel {kernel_file}/{kernel_name}, N={N}")

    def get_all_json_files(self):
        json_files = []
        for root, dirs, files in os.walk(self.subdumpdir):
            for file in files:
                if file.endswith('.json') and file.startswith('results_all'):
                    json_files.append(os.path.join(root, file))
        return json_files

    def plotgen_all(self):
        cnt = self.get_count()
        if cnt == 0:
            print("No data to plot.")
            return

        # create a figure with cnt subplots
        fig, axs = plt.subplots(cnt, 1, figsize=(10, 5 * cnt))
        fig.tight_layout(pad=3.0)

        if cnt == 1:
            axs = [axs]

        subplot_idx = 0
        for file, kernels in self.data_aggr.items():
            for kernel, N_values in kernels.items():
                for N, data in N_values.items():
                    print(f"Plotting {file}/{kernel} for N={N}")
                    self.plotgen_one(file, kernel, N, axs, fig, subplot_idx)
                    subplot_idx += 1

        # set labels and title
        for i, ax in enumerate(axs):
            ax.set_xlabel('Iteration number')
            ax.set_ylabel('Time (ms)')

        # save the plot in subdump directory
        plt.savefig(f"{self.subdumpdir}/autotuner_convergence.png")

    def plotgen_one(self, k_file, k_kernel, k_n, ax, fig, subplot_idx):
        kernel = self.data_aggr[k_file][k_kernel][k_n]
        # x-axis: iteration number in 'all' list
        # y-axis: time in 'all' list
        x = np.arange(len(kernel['all']))
        y = np.zeros(len(kernel['all']))
        for i, run in enumerate(kernel['all']):
            try :
                y[i] = run['time']
            except:
                y[i] = 0


        ax[subplot_idx].plot(x, y, label=f"{k_kernel} N={k_n} {self.data_aggr[k_file][k_kernel][k_n]['opt']['opt']}")

        # add legend
        ax[subplot_idx].legend()


    def get_count(self):
        cnt = 0
        for file, kernels in self.data_aggr.items():
            for kernel, N_values in kernels.items():
                for N, data in N_values.items():
                    cnt += 1
        return cnt




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subdump', type=str, required=True)
    args = parser.parse_args()
    plotter = GpuAutotunerConvergencePlotter(args.subdump)
    plotter.plotgen_all()
