# exit on failure
set -e

# Check if there is at least one argument supplied
if [ $# -lt 2 ]; then
    echo "Usage: $0 machine_name compiler_exec"
    exit 1
fi

machine=$1
compiler=$2

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
    echo "Processing benchmark $bench_id..."
    cd $bench_id
    if [ "$delete_dumps" = true ]; then
        echo "Deleting dumps for benchmark $bench_id..."
        bash runme.sh --machine=$machine "$compiler" -d
    fi
    echo "Running auto-tune for benchmark $bench_id..."
    bash runme.sh --machine=$machine "$compiler" --auto-tune
    echo "Running final compilation for benchmark $bench_id..."
    bash runme.sh --machine=$machine "$compiler"
    cd ..
}

# Process each benchmark
for bench_id in 2 7 8; do
    process_benchmark $bench_id
done