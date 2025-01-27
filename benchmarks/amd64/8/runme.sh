set -e
current_benchId="8"
source "../../common/runme.matmul.any.sh"
run_benchmark $current_benchId "amd64" "$@"
