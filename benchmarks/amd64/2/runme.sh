set -e
current_benchId="2"
source "../../common/runme.matmul.any.sh"
run_benchmark $current_benchId "amd64" "$@"
