#!/bin/bash

# get the path to the symlinked script (not the real file, the symlink's path)
script_dir=$(dirname "$0")
dump_dir="$script_dir/../../dumps"
dump_dir=$(realpath "$dump_dir")
delete_dumps=false

# Process all arguments: handle -d, compiler, and machine arguments
args=()
machine=""

# First pass to extract machine argument
for i in "$@"; do
    case $i in
        --machine=*)
            machine="${i#*=}"
            ;;
    esac
done

# Check if machine argument was provided
if [ -z "$machine" ]; then
    echo "Error: --machine argument is mandatory"
    echo "Usage: $0 --machine=<string> [-d] [g++|clang++] [extra_flags]"
    exit 1
fi

# Process remaining arguments
for arg in "$@"; do
    if [ "$arg" == "-d" ]; then
        delete_dumps=true
    elif [[ "$arg" =~ ^(g\+\+|clang\+\+) ]]; then
        compiler="$arg"
    elif [[ "$arg" != --machine=* ]]; then  # Explicitly exclude --machine arguments
        args+=("$arg")
    fi
done

# Delete dump directory if -d flag was provided
if [ "$delete_dumps" = true ] && [ -d "$dump_dir" ]; then
  echo "Deleting all contents of $dump_dir"
  rm -rf "$dump_dir"
  mkdir "$dump_dir"
  echo "Deleted all contents of $dump_dir , exiting."
  exit 0
fi

if [ ! -d "$dump_dir" ]; then
  mkdir "$dump_dir"
fi

timestamp=$(date +"%Y%m%d_%H%M%S")
new_dump_dir="$dump_dir/$timestamp"
mkdir "$new_dump_dir"
log_file="$new_dump_dir/output_log_$timestamp.txt"

{
  # Get abs path of sources_main
  sources_main="main.cpp"
  abs_main=$(realpath "$script_dir/$sources_main")
  # Get the dir name of the folder containing sources_main
  benchId=$(basename "$(dirname "$abs_main")")
  echo "sources_main: $sources_main"
  echo "abs_main: $abs_main"
  echo "benchId: $benchId"

  echo "${machine}, ${new_dump_dir}" >> "$dump_dir/benchId${benchId}.txt"

  # run cmake in the build directory for the current directory without cd into it
  cmake -S . -B "$new_dump_dir" -DCMAKE_BUILD_TYPE=Release

  # run make in the build directory for the current directory without cd into it
  make -C "$new_dump_dir" -j

  # run the executable in the build directory for the current directory without cd into it
  (cd "$new_dump_dir" && taskset -c 0 "./output_file") 2>&1
} | tee -a "$log_file"
