#!/bin/bash

compiler=g++

function print_line() {
  echo "############################################"
}

function print_line_long() {
  echo "*"
  echo "*"
  echo "*"
  echo "############################################"
}

script_dir=$(dirname "$(readlink -f "$0")")
dump_dir="$script_dir/../dumps"
delete_dumps=false

# Process all arguments and handle -d flag
args=()
for arg in "$@"; do
  if [ "$arg" == "-d" ]; then
    delete_dumps=true
  else
    args+=("$arg")  # Collect non-`-d` arguments
  fi
done

# Delete dump directory if -d flag was provided
if [ "$delete_dumps" = true ] && [ -d "$dump_dir" ]; then
  echo "Deleting all contents of $dump_dir"
  rm -rf "$dump_dir"
fi

# Check the number of remaining arguments
if [ "${#args[@]}" -lt 1 ] || [ "${#args[@]}" -gt 2 ]; then
  echo "Usage: $0 [-d] <path_to_foo.cpp> [extra_flags]"
  exit 1
fi

source_file=${args[0]}
extra_flags=${args[1]:-}  # Second argument is optional; if not provided, it defaults to an empty string
filename=$(basename "$source_file" .cpp)

if [ ! -d "$dump_dir" ]; then
  mkdir "$dump_dir"
fi

timestamp=$(date +"%Y%m%d_%H%M%S")
new_dump_dir="$dump_dir/$timestamp"
mkdir "$new_dump_dir"
log_file="$new_dump_dir/output_log_$timestamp.txt"

{
  compiler_version=$($compiler --version | head -n 1)
  # native arch is not supported in RISC-V
  flags="-O3 -march=rv64imafdcv1p0 -fno-tree-vectorize -fno-tree-slp-vectorize -Wall -Wextra -v -I$script_dir/../common $extra_flags"
  echo "Arch: RISCV"
  echo "Compiler: $compiler"
  echo "Compiler version: $compiler_version"
  echo "Timestamp: $timestamp"
  echo "Benchmark ID: $filename"
  echo "Source file: $source_file"
  echo "Flags: $flags"
  print_line

  # Compile and capture compiler output
  compile_output=$($compiler $flags -o "$new_dump_dir/$filename" "$source_file" 2>&1)
  compile_status=$?

  if [ $compile_status -eq 0 ]; then
    # Run the binary and capture its output
    (cd "$new_dump_dir" && taskset -c 0 "./$filename") 2>&1
    print_line
    # Print and save compiler output after the binary output
    echo "Compilation succeeded. Log:"
    echo "$compile_output"
    echo "$compile_output" >> "$log_file"
  else
    echo "Compilation failed. Log:"
    echo "$compile_output"
    echo "$compile_output" >> "$log_file"
    exit 1
  fi
} | tee -a "$log_file"
