bash build.riscv.00.sh -d # wipe the dumps dir.

for ((n=64; n<=1024; n*=2)); do
  for ((i=1; i<=64; i*=2)); do
    echo "Benchmarking for Unroll Factor of $i and N of $n."
    bash build.riscv.00.sh "-DUNROLL_FACTOR0=$i -DN=$n"
  done
done
python ../../common/plot.bench_01_02.py --dumps-dir ../../dumps --out "BenchId01.png"