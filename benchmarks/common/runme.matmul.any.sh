#!/bin/bash

set -e

# setup_autotuner_args - Parses command-line arguments for the autotuner setup
#
# Positional Arguments:
#   machine      - The target machine name (mandatory)
#   compiler     - The compiler executable (must be g++ or clang++) (mandatory)
#
# Optional Flags:
#   -d           - Enables deletion of dumps
#   --auto-tune  - Enables auto-tuning
#   --extra      - Additional arguments for tuning (requires a value)
#
setup_autotuner_args() {
    # Check to see if there is at least two positional arguments provided directly from function arguments
    if [[ $# -lt 2 ]]; then
        echo "Error: Missing mandatory arguments for setup_autotuner_args"
        echo "Usage: setup_autotuner_args <machine> <compiler> [-d] [--auto-tune] [--extra <extra>]"
        exit 1
    fi

    machine="$1"
    compiler="$2"
    shift 2

    flag_delete_dumps=0
    flag_auto_tune=0
    extra=""
    is_gcc=0
    is_clang=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -d)
                flag_delete_dumps=1
                shift
                ;;
            --auto-tune)
                flag_auto_tune=1
                shift
                ;;
            --extra)
                extra="$2"
                shift 2
                ;;
            *)
                echo "Error: Invalid argument '$1'"
                exit 1
                ;;
        esac
    done

    # Validate compiler
    if [[ "$compiler" =~ (g\+\+|clang\+\+) ]]; then
        if ! command -v "$compiler" &>/dev/null && [[ ! -x "$compiler" ]]; then
            echo "Error: Compiler '$compiler' not found or not executable."
            exit 1
        fi
    else
        echo "Error: Provided compiler '$compiler' is not a valid g++ or clang++."
        exit 1
    fi

    # Check for GCC or Clang
    if [[ "$compiler" == *"clang++"* ]]; then
        is_clang=1
    elif [[ "$compiler" == *"g++"* ]]; then
        is_gcc=1
    fi

    echo "## Parsed autotuner arguments:"
    echo "## Machine: $machine"
    echo "## Compiler: $compiler"
    echo "## Flag Delete Dumps: $flag_delete_dumps"
    echo "## Flag Auto Tune: $flag_auto_tune"
    echo "## Extra: $extra"
    echo "## Is GCC: $is_gcc"
    echo "## Is Clang: $is_clang"
}

run_benchmark() {
    local current_benchId=$1
    local arch_type=$2
    # Store all remaining arguments in an array
    local script_args=("${@:3}")

    # Validate input parameters
    if [[ -z "$current_benchId" ]] || [[ -z "$arch_type" ]]; then
        echo "Error: Both benchId and architecture type are required"
        echo "Usage: run_benchmark <benchId> <arch_type> [additional arguments...]"
        return 1
    fi

    # Select build script based on architecture type
    local build_script
    case "$arch_type" in
        "amd64")
            build_script="build.amd64.00.sh"
            ;;
        "riscv")
            build_script="build.riscv.00.sh"
            ;;
        *)
            echo "Error: Unsupported architecture type. Use 'amd64' or 'riscv'"
            return 1
            ;;
    esac

    script_dir=$(dirname "$0")
    source "$script_dir/../../common/utils.bash"
    source "$script_dir/../../common/ranges.matmul.sh"

    # Forward all additional arguments to setup_autotuner_args
    setup_autotuner_args "${script_args[@]}"

    # Add new line to the end of the file benchIdXX.txt
    echo "" >> "../../dumps/benchId${current_benchId}.txt"

    # Delete any sub-dumps directories if flag is set
    if [[ "$flag_delete_dumps" -eq 1 ]]; then
        echo "Deleting the dumps directory."
        bash "$build_script" "$machine" "$compiler" -d
        return 0
    fi

    if [[ "$flag_auto_tune" -eq 1 ]]; then
        index=0
        total_benchmarks=$(( ${#range_n[@]} * ${#range_i0[@]} * ${#range_i1[@]} * ${#range_i2[@]} ))

        # amd64 auto-tuning
        if [[ "$arch_type" == "amd64" ]]; then
            echo "Running the autotuner for the amd64 architecture."
            for n in "${range_n[@]}"; do
                for i0 in "${range_i0[@]}"; do
                    for i1 in "${range_i1[@]}"; do
                        for i2 in "${range_i2[@]}"; do
                            index=$((index+1))
                            echo "*** benchmark $index out of $total_benchmarks (percent: $((index*100/total_benchmarks))%)"
                            echo "Percent: $((index*100/total_benchmarks))%, N: $n, Unroll Factors: $i0, $i1, $i2" >> /tmp/progressBenchId${current_benchId}.txt
                            echo "Benchmarking for Unroll Factors: $i0, $i1, $i2 and N of $n."

                            # Define ONLY_RUN_OURS to save time skipping the baseline kernels.
                            bash "$build_script" "$machine" "$compiler" --extra "-DAUTOTUNE_BASELINE_KERNELS -DUNROLL_FACTOR0=$i0 -DUNROLL_FACTOR1=$i1 -DUNROLL_FACTOR2=$i2 -DN=$n -DONLY_RUN_OURS $extra"
                        done
                    done
                done
            done
        fi

        if [[ "$arch_type" == "riscv"  ]]; then
            echo "Running the autotuner for the riscv64 architecture."
            for force_vls in 0 1; do
                for n in "${range_n[@]}"; do
                    for i0 in "${range_i0[@]}"; do
                        for i1 in "${range_i1[@]}"; do
                            for i2 in "${range_i2[@]}"; do
                                index=$((index+1))
                                echo "*** benchmark $index out of $total_benchmarks (percent: $((index*50/total_benchmarks))%)"
                                echo "Percent: $((index*100/total_benchmarks))%, N: $n, Unroll Factors: $i0, $i1, $i2" >> /tmp/progressBenchId${current_benchId}.txt
                                echo "Benchmarking for Unroll Factors: $i0, $i1, $i2 and N of $n."

                                vls_flag=""
                                if [[ "$force_vls" -eq 1 ]]; then
                                    if [[ "$is_gcc" -eq 1 ]]; then
                                        # for gcc
                                        vls_flag="-mrvv-vector-bits=zvl"
                                    elif [[ "$is_clang" -eq 1 ]]; then
                                        # for clang
                                        vls_flag="-mrvv-vector-bits=256"
                                    else
                                        echo "Error: Unsupported compiler."
                                        exit 1
                                    fi
                                fi

                                echo "VLS compiler flag: $vls_flag"

                                # Define ONLY_RUN_OURS to save time skipping the baseline kernels.
                                bash "$build_script" "$machine" "$compiler" --extra "$vls_flag -DAUTOTUNE_BASELINE_KERNELS -DUNROLL_FACTOR0=$i0 -DUNROLL_FACTOR1=$i1 -DUNROLL_FACTOR2=$i2 -DN=$n -DONLY_RUN_OURS $extra"
                            done
                        done
                    done
                done
            done
        fi

    else
        # run with the best configuration found by the autotuner
        for n in "${range_n[@]}"; do
            compiler_version=$($compiler --version | head -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            parse_autotuner_best_conf_json ../../dumps/autotuner.json $current_benchId "$machine" "$compiler_version" $n

            # AUTOTUNE_IS_DISABLED is just a flag that will be reported in the json files by the timer classes.
            # This will help us to find the rows that belong to the run with default tunable parameters.

            # AUTOTUNE_BASELINE_KERNELS enables the baseline kernels (scalar or avx/rvv) auto-tuning.
            echo "Building for N of $n with the auto tuned best config: UNROLL_FACTOR0=$UNROLL_FACTOR0 UNROLL_FACTOR1=$UNROLL_FACTOR1 UNROLL_FACTOR2=$UNROLL_FACTOR2"
            bash "$build_script" "$machine" "$compiler" --best --extra "-DAUTOTUNE_BASELINE_KERNELS -DUNROLL_FACTOR0=$UNROLL_FACTOR0 -DUNROLL_FACTOR1=$UNROLL_FACTOR1 -DUNROLL_FACTOR2=$UNROLL_FACTOR2 -DN=$n $extra"
            echo "Also building for N of $n with the default tunable parameters."
            bash "$build_script" "$machine" "$compiler" --best --extra "-DAUTOTUNE_BASELINE_KERNELS -DAUTOTUNE_IS_DISABLED -DN=$n $extra"
        done
    fi

    # Run the autotuner if the flag is set
    if [[ "$flag_auto_tune" -eq 1 ]]; then
        echo "Running the autotuner."
        python ../../common/python/autotune.py --dumps-dir ../../dumps --benchid $current_benchId
    fi
}