import numpy as np
import kernel_tuner
import json
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import os
import random
import argparse

from exceptiongroup import catch

from autotuner_convergence import GpuAutotunerConvergencePlotter


class MyKernelAutoTuner:
    def __init__(
            self,
            opt: str,
            cuda_capability: str,
            kernel_file: str,
            kernel_name: str,
            is_pot: bool,
            pot_words_per_uint8: int,
            launch_repetitions: int,
            limit_seconds: int,
            limit_iterations: int,
            dumps_dir: str
    ):
        self.opt = opt
        self.cuda_capability = cuda_capability
        self.kernel_file = kernel_file
        self.kernel_name = kernel_name + "<vBM, vBN, vBK, vTM, vTN>"
        self.is_pot = is_pot
        self.pot_words_per_uint8 = pot_words_per_uint8 if is_pot else 1  # enforce 1 for the base kernel
        self.launch_repetitions = launch_repetitions
        self.limit_seconds = limit_seconds
        self.limit_iterations = limit_iterations
        self.dumps_dir = dumps_dir

        # ----------------------------------------
        # these will be set in the autotune method, since they depend on the matrix size
        self.tn_a = None
        self.tn_b = None
        self.tn_b_pot_pack2_uint8 = None
        self.tn_b_pot_pack4_uint8 = None
        self.tn_c = None
        self.tn_gold = None
        self.answer = None
        self.args = None

        # ----------------------------------------
        dbg = False
        if not dbg:
            self.tune_params = dict()
            self.tune_params["block_size_x"] = [128, 256, 512, 1024]
            self.tune_params["block_size_y"] = [1]
            self.tune_params["block_size_z"] = [1]
            self.tune_params["pot_words_per_uint8"] = [self.pot_words_per_uint8] # dont modify this
            self.tune_params["vBM"] = [32, 64, 128]
            self.tune_params["vBN"] = [32, 64, 128]
            self.tune_params["vBK"] = [4]
            self.tune_params["vTM"] = [1,2,4,8]
            self.tune_params["vTN"] = [4,8]
            self.tune_params["vUF0"] = [1,2,4]
            self.tune_params["vUF1"] = [1,2,4,8,32]
            self.tune_params["vUF2"] = [1,2,4,8,32]
            self.tune_params["vUF3"] = [1,2,4]
            self.tune_params["vUF4"] = [1,2,4,8]
            self.tune_params["vUF5"] = [1,2,4,8]
            self.tune_params["vUF6"] = [1,2,4,8]
            self.tune_params["vUF7"] = [1,2,4,8]
            self.tune_params["vUF8"] = [1,2,4,8]
            self.tune_params["vUF9"] = [1,2,4,8]
        else:
            self.tune_params = dict()
            self.tune_params["block_size_x"] = [256]
            self.tune_params["block_size_y"] = [1]
            self.tune_params["block_size_z"] = [1]
            self.tune_params["pot_words_per_uint8"] = [self.pot_words_per_uint8]  # dont modify this
            self.tune_params["vBM"] = [128, ]
            self.tune_params["vBN"] = [128,]
            self.tune_params["vBK"] = [8,]
            self.tune_params["vTM"] = [4,8,]
            self.tune_params["vTN"] = [4,8,]
            self.tune_params["vUF0"] = [1,2]
            self.tune_params["vUF1"] = [1,2]
            self.tune_params["vUF2"] = [1,2]
            self.tune_params["vUF3"] = [1,2]
            self.tune_params["vUF4"] = [1,2]
            self.tune_params["vUF5"] = [1,2]
            self.tune_params["vUF6"] = [1,2]
            self.tune_params["vUF7"] = [1,2]
            self.tune_params["vUF8"] = [1,2]
            self.tune_params["vUF9"] = [1,2]
        # ----------------------------------------
        self.results = {}   # A huge json file
        self.bests = {}     # A brief best only

    @staticmethod
    def validate_kernel_parameters(
            matrix_size: int,
            block_size_flat: int,
            pot_words_per_uint8: int,
            BM: int,
            BN: int,
            BK: int,
            TM: int,
            TN: int) -> tuple[bool, list[str]]:

        """
        Validates the parameters for various CUDA kernels that are supported.
        """

        errors = []

        if matrix_size <= 0:
            errors.append("matrix_size must be positive")

        if BM <= 0 or BN <= 0:
            errors.append("BM and BN must be positive")

        if not (BM & (BM - 1) == 0):
            errors.append("BM should be a power of 2 for optimal performance")

        if not (BN & (BN - 1) == 0):
            errors.append("BN should be a power of 2 for optimal performance")

        if not (TM & (TM - 1) == 0):
            errors.append("TM should be a power of 2 for optimal performance")

        if not (TN & (TN - 1) == 0):
            errors.append("TN should be a power of 2 for optimal performance")

        if not (TN % 4 == 0):
            errors.append("TN should be divisible by 4 for vectorized loads/stores from each thread")

        if BN % 2 != 0:
            errors.append("BN must be divisible by 2 (kernel uses BN/2)")

        if BN % TN != 0:
            errors.append("BN must be divisible by TN")

        if BM % TM != 0:
            errors.append("BM must be divisible by TM")

        if BK % 4 != 0:
            errors.append("BK must be divisible by 4 (kernel loads 4 elements at a time)")

        if BN % 8 != 0:
            errors.append("BN must be divisible by 8 for float4 vectorized loads/stores")

        threads_per_block = (BM * BN) // (TM * TN) // pot_words_per_uint8
        if threads_per_block > 1024:
            errors.append(
                f"Total threads per block ((BM * BN)/(TM * TN)/{pot_words_per_uint8}) must not exceed 1024")
        if threads_per_block < 32:
            errors.append("Total threads per block should be at least 32 (one warp) for efficiency")

        if matrix_size < BM or matrix_size < BN:
            errors.append(f"matrix_size ({matrix_size}) must be >= BM ({BM}) and >= BN ({BN})")

        if BM < 8:
            errors.append("BM must be at least 8 for efficient memory access")
        if BN < 8:
            errors.append("BN must be at least 8 for efficient memory access")

        if TM < 1 or TN < 1:
            errors.append("TM and TN must be positive")
        if TM > BM:
            errors.append("TM cannot be larger than BM")
        if TN > BN:
            errors.append("TN cannot be larger than BN")

        if block_size_flat != (BM * BN) // (TM * TN) // pot_words_per_uint8:
            errors.append(f"block_size_x must be equal to (BM * BN) // (TM * TN) // {pot_words_per_uint8}")

        # print(errors)

        return len(errors) == 0, errors

    @staticmethod
    def restriction_func(conf_dict: dict) -> bool:
        if not MyKernelAutoTuner.validate_kernel_parameters(
                size,
                conf_dict["block_size_x"],
                conf_dict["pot_words_per_uint8"],
                conf_dict["vBM"],
                conf_dict["vBN"],
                conf_dict["vBK"],
                conf_dict["vTM"],
                conf_dict["vTN"]
        )[0]:
            return False

        #local_grid_x = self.custom_grid_x(conf_dict)
        #local_grid_y = self.custom_grid_y(conf_dict)
        # print(f"Valid conf: {conf_dict};;; local_grid_x:{size / local_grid_x} and local_grid_y:{size / local_grid_y} *")
        return True

    @staticmethod
    def custom_grid_x(self, config):
        return config["vBN"]

    @staticmethod
    def custom_grid_y(self, config):
        return config["vBM"]

    @staticmethod
    def verification_func(ref, ans, atol=None):
        # ref is the golden output we supplied to the answer argument.
        # ans is the output of the kernel under auto-tuning.

        maximum_abs_diff = atol if atol is not None else 0.1

        ref = ref[3]  # in the args list, index 3 is the output matrix.
        ans = ans[3]  # in the args list, index 3 is the output matrix.
        diff = np.abs(ref - ans)
        max_diff = np.max(diff)

        if max_diff > maximum_abs_diff:
            print("**** MISMATCH")
            print("UUT:")
            print(f"strides: {ans.strides}")
            rlen = ans.shape[1]
            for i in range(0, rlen):
                print(f"UUT 0,{i}: {ans[0, i]}")

            print("GOLD:")
            print(f"strides: {ref.strides}")
            for i in range(0, rlen):
                print(f"Gold 0,{i}: {ref[0, i]}")

            print(f"Max diff: {max_diff}")
        return max_diff < 0.7

    def autotune(self, matrix_size: int, input_data_dict: dict = None):
        # check if matrix_size is a power of 2 and is a number
        if not isinstance(matrix_size, int):
            raise ValueError("matrix_size must be an integer")
        if matrix_size <= 0:
            raise ValueError("matrix_size must be a positive integer")
        if not (matrix_size & (matrix_size - 1) == 0):
            raise ValueError("matrix_size must be a power of 2")
        # ----------------------------------------
        self.tn_a = input_data_dict['a']
        self.tn_b = input_data_dict['b']
        self.tn_b_pot_pack2_uint8 = input_data_dict['b_pot_pack2_uint8']
        self.tn_b_pot_pack4_uint8 = input_data_dict['b_pot_pack4_uint8']
        self.tn_c = np.zeros_like(self.tn_a)
        self.tn_gold = self.tn_a @ self.tn_b
        self.answer = [None, None, None, self.tn_gold]
        if self.is_pot:
            if self.pot_words_per_uint8 == 2:
                # for matrix_size, u have to use np.int32(), otherwise kernel_tuner will throw an error
                self.args = [np.int32(matrix_size), self.tn_a, self.tn_b_pot_pack2_uint8, self.tn_c]
            elif self.pot_words_per_uint8 == 4:
                self.args = [np.int32(matrix_size), self.tn_a, self.tn_b_pot_pack4_uint8, self.tn_c]
            else:
                raise ValueError("Only 2 and 4 words per uint8 are supported for now")
        else:
            self.args = [np.int32(matrix_size), self.tn_a, self.tn_b, self.tn_c]

        # ----------------------------------------
        results_res, results_env = kernel_tuner.tune_kernel(
            self.kernel_name,
            self.kernel_file,
            (matrix_size, matrix_size),
            self.args,
            self.tune_params,
            restrictions=self.restriction_func,
            grid_div_x=["vBN"],
            grid_div_y=["vBM"],

            #strategy="bayes_opt",
            strategy=self.opt,
            #strategy="bayes_opt",


            iterations=self.launch_repetitions,  # how many times a kernel is run for a given configuration
            compiler_options=[
                "-O3",
                "--fmad=true",
                "--expt-relaxed-constexpr",
                f"-arch=compute_{self.cuda_capability}",
                f"-code=sm_{self.cuda_capability}",
                "-DONLY_KERNELS"
            ],
            strategy_options={
                "time_limit": int(self.limit_seconds),  # in seconds
                "max_fevals": self.limit_iterations,    # maximum number of evaluations
                "maxiter": self.limit_iterations        # maximum number of iterations
            },
            verbose=True,
            answer=self.answer,
            verify=self.verification_func,
            atol=0.7,
            cache=(pathlib.Path(self.dumps_dir) / pathlib.Path(f"cachefile__{self.kernel_file}__{self.kernel_name}__N{matrix_size}.json")).__str__(),
            simulation_mode=False, # Simulates opt from the existing cache file
        )

        # Initialize the nested dictionary structure if it does not exist
        if self.kernel_file not in self.results:
            self.results[self.kernel_file] = {}
            self.bests[self.kernel_file] = {}
        if self.kernel_name not in self.results[self.kernel_file]:
            self.results[self.kernel_file][self.kernel_name] = {}
            self.bests[self.kernel_file][self.kernel_name] = {}
        if matrix_size not in self.results[self.kernel_file][self.kernel_name]:
            self.results[self.kernel_file][self.kernel_name][matrix_size] = {}
            self.bests[self.kernel_file][self.kernel_name][matrix_size] = {}
        self.results[self.kernel_file][self.kernel_name][matrix_size]["best"] = results_env['best_config']
        self.results[self.kernel_file][self.kernel_name][matrix_size]["env"] = results_env
        self.results[self.kernel_file][self.kernel_name][matrix_size]["all"] = results_res
        self.results[self.kernel_file][self.kernel_name][matrix_size]["opt"] = {'opt': self.opt, 'maxiter': self.limit_iterations, 'time_limit': self.limit_seconds}
        self.bests[self.kernel_file][self.kernel_name][matrix_size] = results_env['best_config']
        print(f"Best configuration: {results_env['best_config']}")

    def get_results_all(self):
        return self.results

    def save_results_all(self):
        with open(pathlib.Path(self.dumps_dir) / pathlib.Path(
                f"bests_all__{self.kernel_file}__{self.kernel_name}.json").__str__(), 'w') as fp:
            json.dump(self.bests, fp)
        with open(pathlib.Path(self.dumps_dir) / pathlib.Path(
                f"results_all__{self.kernel_file}__{self.kernel_name}.json").__str__(), 'w') as fp:
            json.dump(self.results, fp)

    def plot_all_times(self):
        # Use self.results to plot the times
        for kernel_file in self.results:
            for kernel_name in self.results[kernel_file]:
                for matrix_size in self.results[kernel_file][kernel_name]:
                    results = self.results[kernel_file][kernel_name][matrix_size]["all"]
                    times = [r["time"] for r in results]
                    # sort times and find the threshold for the least 5% of the times
                    min_time, max_time = np.percentile(times, [5, 95])
                    times_masked = [t for t in times if t <= min_time]

                    plt.figure(figsize=(32, 32))
                    plt.xticks(rotation=90)
                    plt.xticks(sorted(set(times_masked)))
                    sns.histplot(times_masked, bins=100, kde=True)
                    plt.title(f"Kernel: {kernel_name}, Matrix size: {matrix_size}")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Frequency")

                    plt.savefig(pathlib.Path(self.dumps_dir) / pathlib.Path(
                        f"{kernel_file}_{kernel_name}_{matrix_size}.svg").__str__())


def get_matrices_pot4bit_uint8(size):
    """
    Generate matrices for 2-bit and 4-bit PoT representation.
    :param size: The size of the square matrices to be generated.
    :return: A dictionary containing the matrices.
    """
    a = np.random.rand(size, size).astype(np.float32)
    b_exponents = np.random.randint(0, 1 + 1, (size, size)).astype(np.int32)
    b_sign = np.random.randint(0, 1 + 1, (size, size)).astype(np.int32)

    assert size % 2 == 0  # 4bit pot
    assert size % 4 == 0  # 2bit pot

    b = np.zeros(shape=(size, size), dtype=np.float32)
    b_pot_pack2_uint8 = np.zeros(shape=(size, size // 2), dtype=np.uint8)
    b_pot_pack4_uint8 = np.zeros(shape=(size, size // 4), dtype=np.uint8)

    for j in range(size):
        for i in range(size // 2):
            # little endian, so the first element is the least significant
            s0 = b_sign[j, 2 * i]
            e0 = b_exponents[j, 2 * i]
            s1 = b_sign[j, 2 * i + 1]
            e1 = b_exponents[j, 2 * i + 1]
            b_pot_pack2_uint8[j, i] = s0 << 3 | (e0 & 0b0111)
            b_pot_pack2_uint8[j, i] |= (s1 << 3 | (e1 & 0b0111)) << 4

    for j in range(size):
        for i in range(size // 4):
            # little endian, so the first element is the least significant
            s0 = b_sign[j, 4 * i]
            e0 = b_exponents[j, 4 * i]
            s1 = b_sign[j, 4 * i + 1]
            e1 = b_exponents[j, 4 * i + 1]
            s2 = b_sign[j, 4 * i + 2]
            e2 = b_exponents[j, 4 * i + 2]
            s3 = b_sign[j, 4 * i + 3]
            e3 = b_exponents[j, 4 * i + 3]

            b_pot_pack4_uint8[j, i] = 0
            b_pot_pack4_uint8[j, i] |= (s0 << 1 | (e0 & 0b1)) << 0
            b_pot_pack4_uint8[j, i] |= (s1 << 1 | (e1 & 0b1)) << 2
            b_pot_pack4_uint8[j, i] |= (s2 << 1 | (e2 & 0b1)) << 4
            b_pot_pack4_uint8[j, i] |= (s3 << 1 | (e3 & 0b1)) << 6

    for j in range(b.shape[0]):
        for i in range(b.shape[1]):
            b[j, i] = 2 ** b_exponents[j, i]
            if b_sign[j, i] == 1:
                b[j, i] *= -1

    return {
        'a': a,
        'b_exponents': b_exponents,
        'b_sign': b_sign,
        'b': b,
        'b_pot_pack2_uint8': b_pot_pack2_uint8,
        'b_pot_pack4_uint8': b_pot_pack4_uint8
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # mandatory arguments
    parser.add_argument("cc", type=str, help="cuda_capability, for example: 70 or 89")

    # args with default values
    parser.add_argument("--maxiter", type=int, default=500000, help="Maximum number of iterations for each kernel configuration")
    parser.add_argument("--reps", type=int, default=7, help="Number of repetitions for each kernel configuration")
    parser.add_argument("--time", type=int, default=3600*3, help="Time limit in seconds for each kernel configuration")

    args = parser.parse_args()
    dumps_dir = "../../dumps"
    while True:
        current_time = datetime.datetime.now().strftime("GPU_CUDA_%Y%m%d_%H%M%S")
        random_number = random.randint(1000, 9999)
        unique_dir_name = f"{current_time}_{random_number}"
        sub_dump_dir = pathlib.Path(dumps_dir) / pathlib.Path(unique_dir_name)
        try:
            os.makedirs(sub_dump_dir, exist_ok=False)
            break
        except FileExistsError:
            print(f"Directory {sub_dump_dir} already exists, trying again...")
            continue

    # Print info
    print(f"Unique dump directory: {sub_dump_dir}")
    import pycuda.driver as cuda
    cuda.init()
    device = cuda.Device(0)
    print("CUDA Available Capability:", device.compute_capability())
    print(f"CUDA Requested Capability: {args.cc}")
    print(f"Repetitions: {args.reps}")
    print(f"Max iterations: {args.maxiter}")
    print(f"Time limit: {args.time}")


    autotuners = {}
    for size in [1024,2048,4096]:
        print(f"Generating input tensors for size: {size}")
        matrices = get_matrices_pot4bit_uint8(size)
        for is_pot in [False, True]:
            for words_per_uint8 in [2, 4]:
                # skip if is_pot is False and words_per_uint8 is 4, when is_pot is False, words_per_uint8 is set to 1 internally
                # We don't want to tune the base kernel twice for the same size
                if not is_pot and words_per_uint8 == 4:
                    continue
                k_name = f"KernelMatmulPotUint8Packed{words_per_uint8}" if is_pot else "KernelMatmulBase"
                if k_name not in autotuners:
                    autotuners[k_name] = MyKernelAutoTuner(
                        # basinhopping, bayes_opt, brute_force, minimize, dual annealing, diff_evo, firefly_algorithm,
                        # genetic_algorithm, greedy_ils, greedy_mls, mls, ordered_greedy_mls, pso, random_sample,
                        # simulated_annealing
                        opt="firefly_algorithm",
                        cuda_capability=args.cc,
                        kernel_file="kernels.cu",
                        kernel_name=k_name,
                        is_pot=is_pot,
                        pot_words_per_uint8=words_per_uint8 if is_pot else 1,
                        launch_repetitions=args.reps,
                        limit_seconds=args.time,
                        limit_iterations=args.maxiter,
                        dumps_dir=pathlib.Path(sub_dump_dir).__str__()
                    )
                    autotuners[k_name].save_results_all()
                    GpuAutotunerConvergencePlotter(sub_dump_dir).plotgen_all()

                print(f"Starting autotuning for {k_name}, size: {size}")
                now = datetime.datetime.now()
                autotuners[k_name].autotune(size, input_data_dict=matrices)
                print(f"Finished autotuning for {k_name},  size: {size}, took: {datetime.datetime.now() - now}")

    for autotuner in autotuners.values():
        autotuner.save_results_all()
        GpuAutotunerConvergencePlotter(sub_dump_dir).plotgen_all()
        #autotuner.plot_all_times()
