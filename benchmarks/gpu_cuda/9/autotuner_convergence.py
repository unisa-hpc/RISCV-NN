import orjson
import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib


class GpuAutotunerConvergencePlotter:
    def __init__(self, subdumpdirs: [str], output_dir: str):
        self.FILE_FORMAT = 'svg'
        self.subdumpdirs = subdumpdirs
        self.output_dir = output_dir
        self.file_paths = self.get_all_json_files()
        self.data_raw = {}
        self.data_aggr = {}
        self.df = pd.DataFrame(columns=['Iteration', 'Kernel', 'N', 'GPU', 'Algorithm', 'Min Time', 'Min Time Global'])

        # file -> kernel -> N -> {
        #                   'best': [..., times:[...], time: 000],
        #                   'env',
        #                   'all': [..., times:[...], time: 000],
        #                   'opt': {'opt: str, 'maxiter': int, 'time_limit': int}
        #                   }

        self.load_data()
        self.aggregate_raw_data()
        self.preprocess_data()

    def load_data(self):
        # open json files and parse them with orjson with try except
        for file in self.file_paths:
            with open(file, 'r') as f:
                try:
                    data = orjson.loads(f.read())
                    self.data_raw[pathlib.Path(file).__str__()] = data
                except Exception as e:
                    print(f"Error reading file {file}: {e}")

    def aggregate_raw_data(self):
        for file, data in self.data_raw.items():
            for kernel_file, kernels in data.items():  # `kernels` contains multiple kernel names
                if kernel_file not in self.data_aggr:
                    self.data_aggr[file] = {}

                for kernel_name, kernel_data in kernels.items():  # Iterate through kernel names
                    if kernel_name not in self.data_aggr[file]:
                        self.data_aggr[file][kernel_name] = {}

                    for N, results in kernel_data.items():  # Iterate through N values
                        if N not in self.data_aggr[file][kernel_name]:
                            self.data_aggr[file][kernel_name][N] = {
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
                            self.data_aggr[file][kernel_name][N]['best'].extend(best_results)
                        else:
                            self.data_aggr[file][kernel_name][N]['best'].append(best_results)

                        if isinstance(all_results, list):
                            self.data_aggr[file][kernel_name][N]['all'].extend(all_results)
                        else:
                            self.data_aggr[file][kernel_name][N]['all'].append(all_results)

                        # Handle 'env' - store only if consistent, otherwise raise a warning
                        if self.data_aggr[file][kernel_name][N]['env'] is None:
                            self.data_aggr[file][kernel_name][N]['env'] = env_value
                        elif self.data_aggr[file][kernel_name][N]['env'] != env_value and env_value is not None:
                            print(f"Warning: Inconsistent 'env' values for kernel {kernel_file}/{kernel_name}, N={N}")

                        # Handle 'opt' - store only if consistent, otherwise raise a warning
                        if self.data_aggr[file][kernel_name][N]['opt'] is None:
                            self.data_aggr[file][kernel_name][N]['opt'] = opt_value
                        elif self.data_aggr[file][kernel_name][N]['opt'] != opt_value and opt_value is not None:
                            print(f"Warning: Inconsistent 'opt' values for kernel {kernel_file}/{kernel_name}, N={N}")

    def preprocess_data(self):
        """
        Preprocesses the data to be used in plotting.
        Constructs the pandas dataframe from self.data_aggr, supporting:
        - Multiple GPUs
        - Multiple Algorithms
        - Multiple Kernels
        - Multiple N values

        self.data_aggr: file -> kernel -> N -> {
                            'best': [..., times:[...], time: 000],
                            'env',
                            'all': [..., times:[...], time: 000],
                            'opt': {'opt: str, 'maxiter': int, 'time_limit': int}
                            }

        """
        # Construct the dataframe from self.data_aggr
        for file, kernels in self.data_aggr.items():
            for kernel, N_values in kernels.items():
                for N, data in N_values.items():
                    print(f"Processing {file}/{kernel} for N={N}")
                    x = np.arange(len(data['all']))
                    y = np.zeros(len(data['all']))

                    for i, run in enumerate(data['all']):
                        try:
                            y[i] = run['time']
                        except:
                            y[i] = -1

                    y = self.extract_pareto_runtime(x, y)
                    for i in range(len(x)):
                        row = {
                            'Iteration': int(x[i]),
                            'Kernel': kernel,
                            'N': int(N),
                            'GPU': data['env']['device_name'],
                            'Algorithm': data['opt']['opt'],
                            'Min Time': y[i],
                            'Min Time Global': min(y)
                        }
                        # add row to dataframe
                        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        print("Preprocessing done.")

    def get_all_json_files(self):
        json_files = []
        for subdumpdir in self.subdumpdirs:
            for root, dirs, files in os.walk(subdumpdir):
                for file in files:
                    if file.endswith('.json') and file.startswith('results_all'):
                        json_files.append(os.path.join(root, file))
        return json_files

    def plotgen_all_in_one_figure2(self):
        # Plots all data in self.df in one figure without any subplots
        fig = plt.figure(figsize=(15, 10))

        unique_N = self.df['N'].unique()

        sns.lineplot(data=self.df, x='Iteration', y='Min Time', hue='Kernel', style='N', markers=False, )
        # logaritmic scale y
        plt.yscale('log')
        plt.title('Autotuner Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Min. Kernel Runtime (ms)')
        plt.savefig(f"{self.output_dir}/autotuner_convergence_all_in_one2.{self.FILE_FORMAT}")

    def plotgen_old_detailed_convergence(self):
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
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Kernel Runtime (ms)')

        # save the plot in subdump directory
        plt.savefig(f"{self.output_dir}/autotuner_convergence.{self.FILE_FORMAT}")

    def plotgen_all(self):
        self.plotgen_old_detailed_convergence()
        for N in self.df['N'].unique():
            self.plotgen2_convergence(fixed_N=N)
            self.plotgen2_convergence2(fixed_N=N)
        self.plotgen2_compare_best()
        self.plotgen2_compare_best2()

    def extract_pareto_runtime(self, data_x, data_y):
        assert len(data_x) == len(data_y)
        # lets just work on data_y
        # we should generate a ndarray of the same size as data_y but with pareto points
        pareto = np.inf
        y_pareto = np.zeros(len(data_y))
        for i in range(len(data_y)):
            if data_y[i] < pareto and data_y[i] > 0:
                pareto = data_y[i]
            y_pareto[i] = pareto
        return y_pareto

    def plotgen_one(self, k_file, k_kernel, k_n, ax, fig, subplot_idx):
        kernel = self.data_aggr[k_file][k_kernel][k_n]
        # x-axis: iteration number in 'all' list
        # y-axis: time in 'all' list
        x = np.arange(len(kernel['all']))
        y = np.zeros(len(kernel['all']))

        for i, run in enumerate(kernel['all']):
            try:
                y[i] = run['time']
            except:
                y[i] = -1

        y = self.extract_pareto_runtime(x, y)

        ax[subplot_idx].plot(x, y, label=f"{k_kernel} N={k_n} {self.data_aggr[k_file][k_kernel][k_n]['opt']['opt']}")
        # Limit y-axis range to min(y) - 10% and min(y) + 20%

        # add legend
        ax[subplot_idx].legend()

        # delete negative values for min/max calculation
        y = y[y > 0]

        y_min = min(y)
        ax[subplot_idx].set_ylim(y_min - 0.02 * y_min, y_min + 0.5 * y_min)

        # log scale
        # ax[subplot_idx].set_yscale('log')

        # add title
        ax[subplot_idx].set_title(f"{k_kernel} N={k_n}, [[Min: {min(y):.3f} ms]], Max: {max(y):.3f} ms")

        for i in range(len(x)):
            row = {
                'Iteration': x[i],
                'Kernel': k_kernel,
                'N': k_n,
                'GPU': kernel['env']['device_name'],
                'Algorithm': self.data_aggr[k_file][k_kernel][k_n]['opt']['opt'],
                'Min Time': y[i],
                'Min Time Global': min(y)
            }
            # add row to dataframe AttributeError: 'DataFrame' object has no attribute 'append'
            self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

    def get_count(self):
        cnt = 0
        for file, kernels in self.data_aggr.items():
            for kernel, N_values in kernels.items():
                for N, data in N_values.items():
                    cnt += 1
        return cnt

    def plotgen2_convergence(self, fixed_N: int):
        """
        Plots one figure with Q subfigures, Q being the number of GPUs.
        Each subfigure contains all the kernels, and algorithms for a specific GPU.
        N is fixed.
        """
        Ns = self.df['N'].unique()
        if fixed_N not in Ns:
            print(f"Error: N={fixed_N} not found in the dataset.")
            print(f"Available N values: {Ns}")
            return

        # Create a figure with Q subplots, Q being the number of GPUs
        Q = len(self.df['GPU'].unique())
        fig, axs = plt.subplots(nrows=Q, ncols=1, figsize=(5, 2 * Q))
        fig.tight_layout(pad=3.0)

        # Get unique GPUs
        unique_GPUs = self.df['GPU'].unique()

        fig.subplots_adjust(top=0.75)

        # Iterate over unique GPUs
        for gpu in unique_GPUs:
            masked_df = self.df[(self.df['GPU'] == gpu) & (self.df['N'] == fixed_N)]
            sns.lineplot(data=masked_df, x='Iteration', y='Min Time',
                         hue='Kernel', style='Algorithm', markers=False, ax=axs[unique_GPUs.tolist().index(gpu)]
            )
            axs[unique_GPUs.tolist().index(gpu)].set_title(f"GPU: {gpu}, N={fixed_N}")
            axs[unique_GPUs.tolist().index(gpu)].set_xlabel('Iteration')
            axs[unique_GPUs.tolist().index(gpu)].set_ylabel('Min Kernel Runtime (ms)')
            axs[unique_GPUs.tolist().index(gpu)].legend()
            # logaritmic scale y
            #axs[unique_GPUs.tolist().index(gpu)].set_yscale('log')

        # remove legend from all subplots
        for ax in axs:
            ax.get_legend().remove()

        # add common legend to the figure
        handles, labels = axs[0].get_legend_handles_labels()

        # Manually change text of the legend entries
        for i, label in enumerate(labels):
            if label.find('Kernel') != -1:
                if label.find('Base') != -1:
                    labels[i] = 'BaseMatmul'
                if label.find('Packed2') != -1:
                    labels[i] = 'FPoT F32:U8:E3:P2'
                if label.find('Packed4') != -1:
                    labels[i] = 'FPoT F32:U8:E1:P4'
            if label.find('firefly') != -1:
                labels[i] = 'FireFly Algorithm'
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize='small')

        # remove x axis label from the first subplot
        axs[0].set_xlabel('')
        axs[0].set_ylabel('')
        axs[1].set_ylabel('')

        # use the same x-axis range for all subplots (union)
        x_min = self.df['Iteration'].min()
        x_max = 1000 #self.df['Iteration'].max()

        _ = self.df[(self.df['N'] == fixed_N)&(self.df['Iteration'] >10)]
        y_min = _['Min Time'].min()
        y_max = _['Min Time'].max()
        for ax in axs:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        fig.text(0.02, 0.5, 'Min. Kernel Runtime (ms)', va='center', rotation='vertical')
        fig.savefig(f"{self.output_dir}/autotuner_convergence_fixed_N_{fixed_N}.{self.FILE_FORMAT}")

    def plotgen2_convergence2(self, fixed_N: int):
        """
        Plots one figure with Q subfigures, Q being the number of GPUs.
        Each subfigure contains all the kernels, and algorithms for a specific GPU.
        N is fixed.
        """
        Ns = self.df['N'].unique()
        if fixed_N not in Ns:
            print(f"Error: N={fixed_N} not found in the dataset.")
            print(f"Available N values: {Ns}")
            return

        # Create a figure with Q subplots, Q being the number of GPUs
        Q = len(self.df['GPU'].unique())
        fig, axs = plt.subplots(nrows=1, ncols=Q, figsize=(2 * Q, 3))
        fig.tight_layout(pad=3.0)

        # Get unique GPUs
        unique_GPUs = self.df['GPU'].unique()

        fig.subplots_adjust(top=0.75)
        sns.set_theme(font_scale=0.75)

        # Iterate over unique GPUs
        for gpu in unique_GPUs:
            masked_df = self.df[(self.df['GPU'] == gpu) & (self.df['N'] == fixed_N)]
            sns.lineplot(data=masked_df, x='Iteration', y='Min Time',
                         hue='Kernel', style='Algorithm', markers=False, ax=axs[unique_GPUs.tolist().index(gpu)]
            )
            gpu_short = ''
            if gpu.find('V100S') != -1:
                gpu_short = 'V100s'
            if gpu.find('Orin') != -1:
                gpu_short = 'Orin Nano'
            axs[unique_GPUs.tolist().index(gpu)].set_title(f"GPU: {gpu_short}, N={fixed_N}")
            axs[unique_GPUs.tolist().index(gpu)].set_xlabel('Iteration')
            axs[unique_GPUs.tolist().index(gpu)].set_ylabel('Min Kernel Runtime (ms)')
            axs[unique_GPUs.tolist().index(gpu)].legend()
            # logaritmic scale y
            #axs[unique_GPUs.tolist().index(gpu)].set_yscale('log')

        # remove legend from all subplots
        for ax in axs:
            ax.get_legend().remove()

        # add common legend to the figure
        handles, labels = axs[0].get_legend_handles_labels()

        # Manually change text of the legend entries
        for i, label in enumerate(labels):
            if label.find('Kernel') != -1:
                if label.find('Base') != -1:
                    labels[i] = 'BaseMatmul'
                if label.find('Packed2') != -1:
                    labels[i] = 'FPoT F32:U8:E3:P2'
                if label.find('Packed4') != -1:
                    labels[i] = 'FPoT F32:U8:E1:P4'
            if label.find('firefly') != -1:
                labels[i] = 'FireFly Algorithm'
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize='small')

        # remove x axis label from the first subplot
        axs[0].set_xlabel('')
        axs[0].set_ylabel('')
        axs[1].set_ylabel('')

        # use the same x-axis range for all subplots (union)
        x_min = self.df['Iteration'].min()
        x_max = 1000 #self.df['Iteration'].max()

        _ = self.df[(self.df['N'] == fixed_N)&(self.df['Iteration'] >10)]
        y_min = _['Min Time'].min()
        y_max = _['Min Time'].max()
        for ax in axs:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        fig.text(0.02, 0.5, 'Min. Kernel Runtime (ms)', va='center', rotation='vertical')
        fig.savefig(f"{self.output_dir}/autotuner_convergence2_fixed_N_{fixed_N}.{self.FILE_FORMAT}")

    def plotgen2_compare_best(self):
        """
        Plots one figure with Q subfigures, Q being the number of GPUs.
        Each subfigure contains all the kernels, and algorithms for a specific GPU.
        N is fixed.
        """
        # Create a figure with Q subplots, Q being the number of GPUs
        Q = len(self.df['GPU'].unique())
        fig, axs = plt.subplots(nrows=Q, ncols=1, figsize=(5, 2 * Q))
        fig.tight_layout(pad=3.0)

        # Get unique GPUs
        unique_GPUs = self.df['GPU'].unique()

        fig.subplots_adjust(top=0.75)

        # Iterate over unique GPUs
        for gpu in unique_GPUs:
            masked_df = self.df[self.df['GPU'] == gpu]
            sns.lineplot(data=masked_df, x='N', y='Min Time Global',
                         hue='Kernel', style='Algorithm', markers=False, ax=axs[unique_GPUs.tolist().index(gpu)]
                         )
            axs[unique_GPUs.tolist().index(gpu)].set_title(f"GPU: {gpu}")
            axs[unique_GPUs.tolist().index(gpu)].set_xlabel('Square Matrix Size (N)')
            axs[unique_GPUs.tolist().index(gpu)].set_ylabel('Autotuned Kernel Runtime (ms)')
            axs[unique_GPUs.tolist().index(gpu)].legend()
            # logaritmic scale y
            # axs[unique_GPUs.tolist().index(gpu)].set_yscale('log')

        # remove legend from all subplots
        for ax in axs:
            ax.get_legend().remove()

        # add common legend to the figure
        handles, labels = axs[0].get_legend_handles_labels()

        # Manually change text of the legend entries
        for i, label in enumerate(labels):
            if label.find('Kernel') != -1:
                if label.find('Base') != -1:
                    labels[i] = 'BaseMatmul'
                if label.find('Packed2') != -1:
                    labels[i] = 'FPoT F32:U8:E3:P2'
                if label.find('Packed4') != -1:
                    labels[i] = 'FPoT F32:U8:E1:P4'
            if label.find('firefly') != -1:
                labels[i] = 'FireFly Algorithm'
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize='small')

        # remove x axis label from the first subplot
        axs[0].set_xlabel('')
        axs[0].set_ylabel('')
        axs[1].set_ylabel('')


        # only have x-ticks on unique Ns
        xticks = self.df['N'].unique()
        for ax in axs:
            ax.set_xticks(np.asarray(xticks, dtype=np.float32))
            # add grid
            ax.grid()

        fig.text(0.02, 0.5, 'Autotuned Kernel Runtime (ms)', va='center', rotation='vertical')
        fig.savefig(f"{self.output_dir}/autotuner_comparison.{self.FILE_FORMAT}")

    def plotgen2_compare_best2(self):
        sns.set_theme(font_scale=0.75)

        g = sns.catplot(
            data=self.df, x='N', y='Min Time Global', kind='bar',
            col='GPU', hue='Kernel', height=3, aspect=0.5
        )

        g.fig.set_size_inches(3, 3)
        g.set(yscale='log')

        # change suptitle fontsize
        g.fig.suptitle('Autotuned Kernel Run-times')

        # Adjust layout to reduce white space
        g.fig.subplots_adjust(left=0.2, right=0.90, top=0.85, bottom=0.23)

        # Rotate x-axis labels
        for ax in g.axes.flat:
            ax.set_xlabel('')  # Ensure xlabel is empty before adding manually
            for label in ax.get_xticklabels():
                label.set_rotation(90)

        # Set correct GPU titles
        for ax in g.axes.flat:
            t = ax.get_title()
            if 'V100S' in t:
                ax.set_title("V100s")
            elif 'Orin' in t:
                ax.set_title("Orin Nano")

        # Fix legend positioning outside plot
        #lgd = g.fig.legend(
        #    loc='upper right', bbox_to_anchor=(1.1, 1), ncol=1, fontsize='small'
        #)
        ## Update legend labels
        #for text in lgd.get_texts():
        #    label = text.get_text()
        #    if 'Base' in label:
        #        text.set_text('BaseMatmul')
        #    elif 'Packed2' in label:
        #        text.set_text('FPoT F32:U8:Pack2')
        #    elif 'Packed4' in label:
        #        text.set_text('FPoT F32:U8:Pack4')
        g._legend.remove()  # Remove default legend

        # set y-axis label
        g.set_ylabels('Runtime (ms)')

        # Add x-axis label centrally
        g.fig.text(0.5, 0.02, 'Square Matrix Size (N)', va='center', ha='center')

        # Save plot
        g.fig.savefig(f"{self.output_dir}/autotuner_comparison2.{self.FILE_FORMAT}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # allow multiple subdump directories
    parser.add_argument('--subdump', type=str, required=False, action='append',
                        help='One sub-dump directory that belongs to a gpu_cuda run')
    parser.add_argument('--output', type=str, required=False, default='/tmp', help='Output directory for the plot')

    args = parser.parse_args()
    plotter = GpuAutotunerConvergencePlotter(args.subdump, args.output)
    plotter.plotgen_all()
