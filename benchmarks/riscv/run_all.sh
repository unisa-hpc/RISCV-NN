#!/bin/bash

#
# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0
#

# exit on failure
set -e
log_file="/tmp/progressRunAllRiscv64_$(date '+%Y%m%d_%H%M%S').log"
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
compilers=("/usr/bin/g++-14" "/usr/bin/clang++-18" "/usr/bin/clang++-17") # GCC13.2 does not support LMUL, so it is not included
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
        bash runme.sh "$machine" "$compiler" -d
    fi
    echo "Running auto-tune for benchmark $bench_id with compiler $compiler..."
    bash runme.sh "$machine" "$compiler" --auto-tune
    echo "Running final compilation for benchmark $bench_id with compiler $compiler..."
    bash runme.sh "$machine" "$compiler"
    cd ..
}

# Process each benchmark
for bench_id in 5 6; do
    for compiler in "${compilers[@]}"; do
        process_benchmark $bench_id "$compiler"
    done
done

echo "Finished running all benchmarks for riscv."
