script_dir=$(dirname "$0")
source "$script_dir/../../common/utils.bash"
setup_autotuner_args "$@"

# Delete any sub-dumps directories of this benchId inside the dumps directory if the flag is set
if [ "$flag_delete_dumps" = true ]; then
  echo "Deleting the dumps directory."
  bash build.amd64.00.sh -d --machine=$machine
  exit 0
fi

for ((n=64; n<=1024; n*=2)); do
  for ((i=1; i<=16; i*=2)); do
    echo "Benchmarking for Unroll Factor of $i and N of $n."
    bash build.amd64.00.sh --machine=$machine "-DUNROLL_FACTOR0=$i -DN=$n $args"
  done
done

# Run the autotuner if the flag is set
if [ "$flag_auto_tune" = true ]; then
  echo "Running the autotuner."
  python ../../common/autotune_find_best_comb.py --dumps-dir ../../dumps --benchid 02 --out ../../dumps/benchId02_autotuned.txt
fi

# Run the plotting script if the auto-tuner flag is not set
if [ "$flag_auto_tune" = false ]; then
  echo "Running the plotting script."
  python ../../common/plot.bench_01_02.py --dumps-dir ../../dumps --out "BenchId02.png"
fi
