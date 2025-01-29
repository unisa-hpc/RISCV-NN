# exit on failure
set -e

# Check if there is at least one argument supplied
if [ $# -lt 1 ]; then
    echo "Usage: $0 machine_name"
    exit 1
fi

mkdir -p ../dumps
machine=$1

# Array of compiler executable names
compilers=("/usr/bin/g++-14" "/usr/bin/g++-13" "/usr/bin/clang++-18" "/usr/bin/clang++-17")
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

#pip install --user argparse pandas colorama pathlib matplotlib numpy seaborn
#sudo apt install python3-pandas python3-colorama

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
for bench_id in 1 5 6; do
    for compiler in "${compilers[@]}"; do
        process_benchmark $bench_id "$compiler"
    done
done
