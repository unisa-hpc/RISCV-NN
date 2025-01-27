# This script runs the benchmark for the current benchId.
# It runs the benchmark for different values of N and Unroll Factor.
# It also runs the autotuner if the flag is set.
# It runs the plotting script if the auto-tuner flag is not set.
# It deletes the sub-dumps directories of this benchId inside the dumps directory if the flag is set.
# Arguments:
#   -d: Delete the sub-dumps directories of this benchId inside the dumps directory.
#   --auto-tune: Run the autotuner.
#   --machine: The machine name.
#   Rest of the arguments are passed to the build script.
#
# Example usage:
#   ./runme.sh --machine=aws_c5 -d
#   ./runme.sh --machine=aws_c5 --auto-tune
#
# The last command will run the autotuner and show the best configuration in terms of median runtime for
# any unique (N, Machine) pair.

current_benchId="05"

script_dir=$(dirname "$0")
source "$script_dir/../../common/utils.bash"
source "$script_dir/../../common/ranges.matmul.sh"
setup_autotuner_args "$@"

# Add new line to the end of the file benchIdXX.txt
echo "" >> "../../dumps/benchId${current_benchId}.txt"

# Delete any sub-dumps directories of this benchId inside the dumps directory if the flag is set
if [ "$flag_delete_dumps" = true ]; then
  echo "Deleting the dumps directory."
  bash build.riscv.00.sh -d --machine=$machine "$compiler"
  exit 0
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
          echo "Benchmarking for Unroll Factor of $i and N of $n."
          bash build.riscv.00.sh --machine=$machine "$compiler" "-DUNROLL_FACTOR0=$i0 -DUNROLL_FACTOR1=$i1 -DUNROLL_FACTOR2=$i2 -DN=$n $args"
        done
      done
    done
  done
else
  # Rename the old benchIdXX.txt file to benchIdXX_autotune.txt
  # We keep `autotuner.json` created by the python script as is; We need it
  mv "../../dumps/benchId${current_benchId}.txt" "../../dumps/benchId${current_benchId}_autotune.txt"

  for n in "${range_n[@]}"; do
    # parse the autotuner json file and get the best configuration for this N
    compiler_version=$($compiler --version | head -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    parse_autotuner_best_conf_json ../../dumps/autotuner.json $current_benchId "$machine" "$compiler_version" $n
    echo "Building for N of $n with the auto tuned best config: UNROLL_FACTOR0=$UNROLL_FACTOR0 UNROLL_FACTOR1=$UNROLL_FACTOR1 UNROLL_FACTOR2=$UNROLL_FACTOR2"
    bash build.riscv.00.sh --machine=$machine "$compiler" "-DUNROLL_FACTOR0=$UNROLL_FACTOR0 -DUNROLL_FACTOR1=$UNROLL_FACTOR1 -DUNROLL_FACTOR2=$UNROLL_FACTOR2 -DN=$n $args"
    echo "Also building for N of $n with the default tunable parameters."
    bash build.riscv.00.sh --machine=$machine "$compiler" "-DN=$n $args"
  done
fi

# Run the autotuner if the flag is set
if [ "$flag_auto_tune" = true ]; then
  echo "Running the autotuner."
  python ../../common/python/autotune.py --dumps-dir ../../dumps --benchid $current_benchId
fi

# Run the plotting script if the auto-tuner flag is not set
if [ "$flag_auto_tune" = false ]; then
  echo "Running the plotting script."
  python ../../common/plot.runtimes.per.benchId.py --dumps-dir "../../dumps" --benchid "$current_benchId" --out "../../dumps/BenchId${current_benchId}.png"
fi
