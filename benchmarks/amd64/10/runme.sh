set -e
current_benchId="10"
source "../../common/runme.matmul.any.sh"
run_benchmark $current_benchId "amd64" "$@"
