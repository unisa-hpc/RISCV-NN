bash build.amd64.00.sh -d # wipe the dumps dir.

for ((i=1; i<=64; i*=2)); do
  echo "Benchmarking for Unroll Factor of $i ."
  bash build.amd64.00.sh 02.cpp "-DUNROLL_FACTOR0=$i"
done

python _plot_per_unrollfactors.py --dumps-dir ../../dumps