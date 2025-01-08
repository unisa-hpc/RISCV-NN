#bash build.amd64.00.sh -d # wipe the dumps dir.

for ((n=64; n<=1024; n*=2)); do
  for ((i=1; i<=16; i*=2)); do
    echo "Benchmarking for Unroll Factor of $i and N of $n."
    bash build.amd64.00.sh "-DUNROLL_FACTOR0=$i -DN=$n" "$@"
  done
done
python ../../common/autotune_find_best_comb.py --dumps-dir ../../dumps --benchid 02 --out ../../dumps/benchId02_autotuned.txt
python ../../common/plot.bench_01_02.py --dumps-dir ../../dumps --out "BenchId02.png"