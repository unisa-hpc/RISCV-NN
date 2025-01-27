set -e
current_benchId="5"
source "../../common/runme.matmul.any.sh"
run_benchmark $current_benchId "riscv" "$@"
