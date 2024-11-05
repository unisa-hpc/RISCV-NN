#!/bin/bash

compiler=g++

function sepertor() {
  echo "*"
  echo "*"
  echo "*"
  echo "========================================"
}

script_dir=$(dirname "$(readlink -f "$0")")
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <path_to_foo.cpp>"
  exit 1
fi
source_file=$1
filename=$(basename "$source_file" .cpp)
dump_dir="$script_dir/../dumps"
if [ ! -d "$dump_dir" ]; then
  mkdir "$dump_dir"
fi
timestamp=$(date +"%Y%m%d_%H%M%S")
new_dump_dir="$dump_dir/$timestamp"
mkdir "$new_dump_dir"
log_file="$new_dump_dir/output_log_$timestamp.txt"

{
  compiler_version=$($compiler --version | head -n 1)
  flags="-O3 -mavx2 -fno-tree-vectorize -fno-tree-slp-vectorize -Wall -Wextra -v -I$script_dir/../common"
  echo "Arch: AMD64"
  echo "Compiler: $compiler"
  echo "Compiler version: $compiler_version"
  echo "Timestamp: $timestamp"
  echo "Benchmark ID: $filename"
  echo "Source file: $source_file"
  echo "Flags: $flags"
  sepertor

  $compiler $flags -o "$new_dump_dir/$filename" "$source_file" 2>&1
  sepertor
  if [ $? -eq 0 ]; then
    "taskset -c 0 $new_dump_dir/$filename" 2>&1
  else
    sepertor
    echo "Compilation failed."
    exit 1
  fi
} | tee "$log_file"