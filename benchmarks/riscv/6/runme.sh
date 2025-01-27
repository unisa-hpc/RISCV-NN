set -e
current_benchId="6"
source "../../common/runme.matmul.any.sh"
run_benchmark $current_benchId "riscv" "$@"
