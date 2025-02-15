import numpy as np
import kernel_tuner
import json
import prettyprinter



def get_matrices_pot4bit_uint8(size):
    """
    Generate matrices for 2-bit and 4-bit PoT representation.
    :param size: The size of the square matrices to be generated.
    :return: A dictionary containing the matrices.
    """
    a = np.random.rand(size, size).astype(np.float32)
    b_exponents = np.random.randint(0, 1+1, (size, size)).astype(np.int32)
    b_sign = np.random.randint(0, 1+1, (size, size)).astype(np.int32)

    assert size % 2 == 0 # 4bit pot
    assert size % 4 == 0 # 2bit pot

    b = np.zeros(shape=(size, size), dtype=np.float32)
    b_pot_pack2_uint8 = np.zeros(shape=(size, size//2), dtype=np.uint8)
    b_pot_pack4_uint8 = np.zeros(shape=(size, size//4), dtype=np.uint8)

    for j in range(size):
        for i in range(size // 2):
            # little endian, so the first element is the least significant
            s0 = b_sign[j, 2*i]
            e0 = b_exponents[j, 2*i]
            s1 = b_sign[j, 2*i+1]
            e1 = b_exponents[j, 2*i+1]
            b_pot_pack2_uint8[j, i] = s0<<3 | (e0 & 0b0111)
            b_pot_pack2_uint8[j, i] |= (s1<<3 | (e1 & 0b0111))<<4

    for j in range(size):
        for i in range(size // 4):
            # little endian, so the first element is the least significant
            s0 = b_sign[j, 4*i]
            e0 = b_exponents[j, 4*i]
            s1 = b_sign[j, 4*i+1]
            e1 = b_exponents[j, 4*i+1]
            s2 = b_sign[j, 4*i+2]
            e2 = b_exponents[j, 4*i+2]
            s3 = b_sign[j, 4*i+3]
            e3 = b_exponents[j, 4*i+3]

            b_pot_pack4_uint8[j, i] = 0
            b_pot_pack4_uint8[j, i] |= (s0<<1 | (e0 & 0b1))<<0
            b_pot_pack4_uint8[j, i] |= (s1<<1 | (e1 & 0b1))<<2
            b_pot_pack4_uint8[j, i] |= (s2<<1 | (e2 & 0b1))<<4
            b_pot_pack4_uint8[j, i] |= (s3<<1 | (e3 & 0b1))<<6

    for j in range(b.shape[0]):
        for i in range(b.shape[1]):
            b[j, i] = 2**b_exponents[j, i]
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

# The size of our vectors that we pass to our vector add kernel
size = 128

matrices = get_matrices_pot4bit_uint8(size)
a = matrices['a']
b = matrices['b']
b_pot_pack2_uint8 = matrices['b_pot_pack2_uint8']
b_pot_pack4_uint8 = matrices['b_pot_pack4_uint8']
c = np.zeros_like(a)
n = np.int32(size)
gold = a @ b

# Now we combine these variables in an argument list, which matches
# the order and types of the function arguments of our GPU kernel
args = [n, a, b_pot_pack2_uint8, c]

# The next step is to create a dictionary to tell Kernel Tuner about
# the tunable parameters in our code and what values these may take
tune_params = dict()
tune_params["block_size_x"] = [32, 64, 128, 256, 512, 1024]
tune_params["block_size_y"] = [1]
tune_params["block_size_z"] = [1]
tune_params["vBM"] = [128]
tune_params["vBN"] = [128]
tune_params["vBK"] = [8]
tune_params["vTM"] = [8]
tune_params["vTN"] = [8]
tune_params["vUF0"] = [1,]
tune_params["vUF1"] = [1,]
tune_params["vUF2"] = [1,]
tune_params["vUF3"] = [1,]
tune_params["vUF4"] = [1,2,3,4,5,6,7,8]
tune_params["vUF5"] = [1,2,3,4,5,6,7,8]
tune_params["vUF6"] = [1,2,3,4,5,6,7,8]
tune_params["vUF7"] = [1,2,3,4,5,6,7,8]
tune_params["vUF8"] = [1,]
tune_params["vUF9"] = [1,]



def validate_kernel_parameters(matrix_size: int, block_size_flat: int, BM: int, BN: int, BK: int, TM: int, TN: int) -> tuple[bool, list[str]]:
    """
    Validates the parameters for the KernelMatMul08_PoT4bits_As_Colmajor_vecloadstore CUDA kernel.
    """
    errors = []

    # 1. Basic size constraints
    if matrix_size <= 0:
        errors.append("matrix_size must be positive")

    # 2. Fixed parameter constraints
    if BK != 8:
        errors.append("BK must be 8 (fixed in launcher code)")

    # 3. Block size constraints
    if BM <= 0 or BN <= 0:
        errors.append("BM and BN must be positive")

    if not (BM & (BM - 1) == 0):
        errors.append("BM should be a power of 2 for optimal performance")

    if not (BN & (BN - 1) == 0):
        errors.append("BN should be a power of 2 for optimal performance")

    # 4. Thread tile constraints
    if not (TM & (TM - 1) == 0):
        errors.append("TM should be a power of 2 for optimal performance")

    if not (TN & (TN - 1) == 0):
        errors.append("TN should be a power of 2 for optimal performance")

    # 5. Divisibility constraints (from kernel implementation)
    if BN % 2 != 0:
        errors.append("BN must be divisible by 2 (kernel uses BN/2)")

    if BN % TN != 0:
        errors.append("BN must be divisible by TN")

    if BM % TM != 0:
        errors.append("BM must be divisible by TM")

    if BK % 4 != 0:
        errors.append("BK must be divisible by 4 (kernel loads 4 elements at a time)")

    # 6. Vector load/store constraints
    if BN % 8 != 0:
        errors.append("BN must be divisible by 8 for float4 vectorized loads/stores")

    # 7. Thread block size constraints
    threads_per_block = (BM * BN) // (TM * TN) // 2
    if threads_per_block > 1024:
        errors.append("Total threads per block ((BM * BN)/(TM * TN)/2) must not exceed 1024")
    if threads_per_block < 32:
        errors.append("Total threads per block should be at least 32 (one warp) for efficiency")

    # 8. Shared memory constraints
    shared_mem_size = (BK * BM) * 4  # As[] array (4 bytes per uint32_t)
    shared_mem_size += BK * (BN//2 + 5)  # Bs[] array (1 byte per uint8_t)
    if shared_mem_size > 48 * 1024:  # 48KB is typical max shared memory per block
        errors.append(f"Shared memory usage ({shared_mem_size} bytes) exceeds 48KB")

    # 9. Matrix size constraints
    if matrix_size < BM or matrix_size < BN:
        errors.append(f"matrix_size ({matrix_size}) must be >= BM ({BM}) and >= BN ({BN})")

    # 10. Minimum size constraints for vectorized operations
    if BM < 8:
        errors.append("BM must be at least 8 for efficient memory access")
    if BN < 8:
        errors.append("BN must be at least 8 for efficient memory access")

    # 11. Thread tile size constraints
    if TM < 1 or TN < 1:
        errors.append("TM and TN must be positive")
    if TM > BM:
        errors.append("TM cannot be larger than BM")
    if TN > BN:
        errors.append("TN cannot be larger than BN")

    if block_size_flat != (BM * BN) // (TM * TN) // 2:
        errors.append("block_size_x must be equal to (BM * BN) // (TM * TN) // 2")

    return len(errors) == 0, errors

def restriction_func_ORIG(conf_dict: dict) -> bool:
    usable_block_sizes = int(np.ceil((conf_dict["vBM"] * conf_dict["vBN"])/(conf_dict["vTM"] * conf_dict["vTN"]*2.0)))
    #(conf_dict["vBM"] * conf_dict["vBN"] + (conf_dict["vTM"] * conf_dict["vTN"]) - 1) // (conf_dict["vTM"] * conf_dict["vTN"] )

    if \
            conf_dict["block_size_x"] != usable_block_sizes or \
                    usable_block_sizes > 1024 or \
                    size % conf_dict["vBM"] != 0 or \
                    size % conf_dict["vBN"] != 0 or \
                    (conf_dict["vBN"] != conf_dict["vBM"] or conf_dict["vBM"] != conf_dict["vBK"] or conf_dict["vBN"] != conf_dict["vBK"]) or \
                    conf_dict["vTM"] != conf_dict["vTN"]:
        return False

    local_grid_x = custom_grid_x(conf_dict)
    local_grid_y = custom_grid_y(conf_dict)
    print(f"Valid1 usable_block_sizes:{usable_block_sizes} and block_size_x:{conf_dict['block_size_x']}")
    print(f"Valid2 local_grid_x:{size/local_grid_x} and local_grid_y:{size/local_grid_y}")
    return True


def restriction_func(conf_dict: dict) -> bool:
    if validate_kernel_parameters(size, conf_dict["block_size_x"], conf_dict["vBM"], conf_dict["vBN"], conf_dict["vBK"], conf_dict["vTM"], conf_dict["vTN"])[0] == False:
        return False

    local_grid_x = custom_grid_x(conf_dict)
    local_grid_y = custom_grid_y(conf_dict)
    print(f"Valid conf: {conf_dict};;; local_grid_x:{size/local_grid_x} and local_grid_y:{size/local_grid_y} *")
    return True

def verification_func(ref, ans, atol=None):
    # ref is the golden output we supplied to the answer argument.
    # ans is the output of the kernel under auto-tuning.
    ref = ref[3] # in the args list, index 3 is the output matrix.
    ans = ans[3] # in the args list, index 3 is the output matrix.
    diff = np.abs(ref - ans)
    max_diff = np.max(diff)

    print("UUT:")
    print(f"strides: {ans.strides}")
    for i in range(0, size):
        print(f"UUT 0,{i}: {ans[0, i]}")

    print("GOLD:")
    print(f"strides: {ref.strides}")
    for i in range(0, size):
        print(f"Gold 0,{i}: {ref[0, i]}")

    print("b_pot_pack2_uint8:")
    print(f"strides: {b_pot_pack2_uint8.strides}")
    for i in range(0, size//2):
        print(f"b_pot_pack2_uint8 0,{i}: {b_pot_pack2_uint8[0, i]}")

    print(f"Max diff: {max_diff}")
    return max_diff < 0.7

def custom_grid_x(config):
    return config["vBN"]

def custom_grid_y(config):
    return config["vBM"]


answer = [None, None, None, gold]
res, env = kernel_tuner.tune_kernel(
    "KernelMatmulPotUint8Packed2<vBM, vBN, vBK, vTM, vTN>",
    "kernels.cu",
    (size, size),
    args,
    tune_params,
    restrictions=restriction_func,

    #grid_div_x=custom_grid_x,
    #grid_div_y=custom_grid_y,
    #grid_div_z=None,

    grid_div_x=["vBN"],
    grid_div_y=["vBM"],

    strategy="bayes_opt",
    iterations=5, # Numnber of times a kenel is run for a given configuration
    compiler_options=[
        "-O3",
        "-Xptxas=\"-v\"",
        "--fmad=true",
        "--expt-relaxed-constexpr",
        "-arch=compute_89", # 70 for v100s
        "-code=sm_89",
        "-DONLY_KERNELS"
    ],
    strategy_options={
        "time_limit": int(15),  # in seconds
        "max_fevals": 100000    # maximum number of evaluations
    },
    verbose=True,
    answer=answer,
    verify=verification_func,
    atol=0.7
)
with open("results.json", 'w') as fp:
    json.dump(res, fp)

with open("results_env.json", 'w') as fp:
    json.dump(env, fp)

first_config = min(res, key=lambda x:x['time'])
print(f"Best configuration: {first_config}")
