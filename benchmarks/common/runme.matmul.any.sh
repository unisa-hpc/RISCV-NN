#!/bin/bash

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
    if [ "$flag_delete_dumps" = true ]; then
        echo "Deleting the dumps directory."
        bash "$build_script" -d --machine=$machine "$compiler"
        return 0
    fi

    if [ "$flag_auto_tune" = true ]; then
        index=0
        total_benchmarks=$(( ${#range_n[@]} * ${#range_i0[@]} * ${#range_i1[@]} * ${#range_i2[@]} ))

        for n in "${range_n[@]}"; do
            for i0 in "${range_i0[@]}"; do
                for i1 in "${range_i1[@]}"; do
                    for i2 in "${range_i2[@]}"; do
                        index=$((index+1))
                        echo "*** benchmark $index out of $total_benchmarks (percent: $((index*100/total_benchmarks))%)"
                        echo "Percent: $((index*100/total_benchmarks))%, N: $n, Unroll Factors: $i0, $i1, $i2" >> /tmp/progressBenchId${current_benchId}.txt
                        echo "Benchmarking for Unroll Factors: $i0, $i1, $i2 and N of $n."
                        bash "$build_script" --machine=$machine "$compiler" "-DAUTOTUNE_BASELINE_KERNELS -DUNROLL_FACTOR0=$i0 -DUNROLL_FACTOR1=$i1 -DUNROLL_FACTOR2=$i2 -DN=$n $args"
                    done
                done
            done
        done
    else
        for n in "${range_n[@]}"; do
            compiler_version=$($compiler --version | head -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            parse_autotuner_best_conf_json ../../dumps/autotuner.json $current_benchId "$machine" "$compiler_version" $n
            echo "Building for N of $n with the auto tuned best config: UNROLL_FACTOR0=$UNROLL_FACTOR0 UNROLL_FACTOR1=$UNROLL_FACTOR1 UNROLL_FACTOR2=$UNROLL_FACTOR2"
            bash "$build_script" --machine=$machine --best "$compiler" "-DAUTOTUNE_BASELINE_KERNELS -DUNROLL_FACTOR0=$UNROLL_FACTOR0 -DUNROLL_FACTOR1=$UNROLL_FACTOR1 -DUNROLL_FACTOR2=$UNROLL_FACTOR2 -DN=$n $args"
            echo "Also building for N of $n with the default tunable parameters."
            bash "$build_script" --machine=$machine --best "$compiler" "-DAUTOTUNE_BASELINE_KERNELS -DN=$n $args"
        done
    fi

    # Run the autotuner if the flag is set
    if [ "$flag_auto_tune" = true ]; then
        echo "Running the autotuner."
        python ../../common/python/autotune.py --dumps-dir ../../dumps --benchid $current_benchId
    fi
}