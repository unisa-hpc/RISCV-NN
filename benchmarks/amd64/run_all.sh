# exit on failure
set -e

log_file="/tmp/progressRunAllAmd64_$(date '+%Y%m%d_%H%M%S').log"
exec &> >(tee -a "$log_file")

# Check if there is at least one argument supplied
if [ $# -lt 1 ]; then
    echo "Usage: $0 machine_name"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "jq could not be found. Please install jq."
    exit 1
fi

mkdir -p ../dumps
machine=$1

# Array of compiler executable names
# Initialize arrays for compilers' paths
declare -a compilers=()

# Clang++ compilers
spack load llvm@18.1.8
compilers+=("$(spack location -i llvm@18.1.8)/bin/clang++")
spack load llvm@17.0.6
compilers+=("$(spack location -i llvm@17.0.6)/bin/clang++")

# G++ compilers
spack load gcc@14.2.0
compilers+=("$(spack location -i gcc@14.2.0)/bin/g++")
spack load gcc@13.2.0
compilers+=("$(spack location -i gcc@13.2.0)/bin/g++")

for i in "${!compilers[@]}"; do
  echo "** Index: $i, Compiler Path: ${compilers[$i]}"
done

# Prompt user about deleting the dumps directory
read -p "Do you want to delete the related sub-dumps directories related to each benchId? (y/n): " delete_dumps_input
if [[ "$delete_dumps_input" =~ ^[Yy]$ ]]; then
    delete_dumps=true
else
    delete_dumps=false
fi

pip install --user argparse pandas colorama pathlib matplotlib numpy seaborn

# Function to process each benchmark
process_benchmark() {
    local bench_id=$1
    local compiler=$2
    echo "Processing benchmark $bench_id with compiler $compiler..."
    cd $bench_id
    if [ "$delete_dumps" = true ]; then
        echo "Deleting dumps for benchmark $bench_id..."
        bash runme.sh --machine=$machine "$compiler" -d
    fi
    echo "Running auto-tune for benchmark $bench_id with compiler $compiler..."
    bash runme.sh --machine=$machine "$compiler" --auto-tune
    echo "Running final compilation for benchmark $bench_id with compiler $compiler..."
    bash runme.sh --machine=$machine "$compiler"
    cd ..
}

# Process each benchmark
for bench_id in 2 7 8; do
    for compiler in "${compilers[@]}"; do
        process_benchmark $bench_id "$compiler"
    done
done
