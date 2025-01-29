#!/bin/bash

set -e
compiler=g++

# Dynamically set compiler flags based on compiler type
function set_compiler_flags() {
    local compiler_path="$1"
    local extra_flags="$2"
    echo "Compiler passed to set_compiler_flags: $compiler_path"

    # Extract compiler basename from path
    local compiler_name=$(basename "$compiler_path")

    # G++ flags
    if [[ "$compiler_name" =~ ^g\+\+(|-[0-9]+([.][0-9]+)*)$ ]]; then
        echo "Using G++ compatible flags."
        flags_main="-O3 -march=native -fno-tree-vectorize -fno-tree-slp-vectorize ${new_dump_dir}/libvec.a ${new_dump_dir}/libscalarvec.a ${new_dump_dir}/libscalarnovec.a -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_vec="-c -O3 -march=native -fno-tree-vectorize -fno-tree-slp-vectorize -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_scalar_vec="-c -O3 -march=native -DAUTOVEC -fopt-info-vec-all -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_scalar_novec="-c -O3 -march=native -fno-tree-vectorize -fno-tree-slp-vectorize -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
    elif [[ "$compiler_name" =~ ^clang\+\+(|-[0-9]+([.][0-9]+)*)$ ]]; then
        echo "Using Clang++ compatible flags."
        flags_main="-O3 -march=native -fno-vectorize ${new_dump_dir}/libvec.a ${new_dump_dir}/libscalarvec.a ${new_dump_dir}/libscalarnovec.a -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_vec="-c -O3 -march=native -fno-vectorize -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_scalar_vec="-c -O3 -march=native -DAUTOVEC -fvectorize -Rpass=loop-vectorize -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_scalar_novec="-c -O3 -march=native -fno-vectorize -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
    else
        echo "Error: Unrecognized compiler '$compiler_name'. Must be g++ or clang++ (with optional version number)"
        exit 1
    fi
}

# get the path to the symlinked script (not the real file, the symlink's path)
script_dir=$(dirname "$0")
dump_dir="$script_dir/../../dumps"
dump_dir=$(realpath "$dump_dir")
delete_dumps=false
source "$script_dir/../../common/utils.bash"

# Process all arguments: handle -d, compiler, and machine arguments
machine=""
best=false  # Add flag for --best option
extra_flags=""

# Process all arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --machine=*)
            machine="${1#*=}"
            shift
            ;;
        --best)
            best=true
            shift
            ;;
        -d)
            delete_dumps=true
            shift
            ;;
        -D*|[-+][A-Za-z]*|-[A-Za-z]*=[A-Za-z0-9]*)
            # Handle compiler flags that start with -D or other common flag patterns
            if [[ -z "$extra_flags" ]]; then
                extra_flags="$1"
            else
                extra_flags="$extra_flags $1"
            fi
            shift
            ;;
        *)
            # Check if the argument is a compiler path
            if [[ -x "$1" ]] && [[ $(basename "$1") =~ ^(g\+\+|clang\+\+)(-[0-9]+([.][0-9]+)*)?$ ]]; then
                if command -v "$1" &> /dev/null; then
                    compiler="$1"
                else
                    echo "Error: Compiler '$1' not found"
                    exit 1
                fi
            else
                # If not a compiler and extra_flags is empty, treat as extra flags
                if [[ -z "$extra_flags" ]]; then
                    extra_flags="$1"
                else
                    extra_flags="$extra_flags $1"
                fi
            fi
            shift
            ;;
    esac
done

# Check if machine argument was provided
if [ -z "$machine" ]; then
    echo "Error: --machine argument is mandatory"
    echo "Usage: $0 --machine=<string> [--best] [-d] [/path/to/g++|/path/to/clang++] [extra_flags]"
    echo "Required: machine"
    echo "Optional: --best, -d, path to g++/clang++ (with optional suffix), extra_flags"
    exit 1
fi

sources_main="main.cpp"
sources_vec="vectorized.cpp"
sources_scalar_vec="scalar.cpp"
sources_scalar_novec="scalar.cpp"

if [ ! -d "$dump_dir" ]; then
  mkdir "$dump_dir"
fi

timestamp=$(date +"%Y%m%d_%H%M%S")-${RANDOM}
new_dump_dir="$dump_dir/$timestamp"
mkdir "$new_dump_dir"
log_file="$new_dump_dir/output_log_$timestamp.txt"

# Get abs path of sources_main
abs_main=$(realpath "$script_dir/$sources_main")
# Get the dir name of the folder containing sources_main
echo "sources_main: $sources_main"
echo "abs_main: $abs_main"
benchId=$(basename "$(dirname "$abs_main")")

if [ "$delete_dumps" = true ]; then
  echo "Deleting..."
  rm -rf "$new_dump_dir"
  delete_flag_handling "$dump_dir/benchId${benchId}.txt" "$dump_dir" "$machine" "$compiler"
fi

# Set compiler flags dynamically
set_compiler_flags "$compiler" "$extra_flags"

{
  compiler_version=$($compiler --version | head -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  echo "Arch: AMD64"
  echo "Compiler: $compiler"
  echo "Compiler version: $compiler_version"
  echo "Timestamp: $timestamp"
  echo "Benchmark ID: $benchId"
  echo "Source file: $sources_main $sources_vec $sources_scalar_vec $sources_scalar_novec"
  echo "flags_main: $flags_main"
  echo "flags_vec: $flags_vec"
  echo "flags_scalar_vec: $flags_scalar_vec"
  echo "flags_scalar_novec: $flags_scalar_novec"
  print_line

  # Compile and capture compiler output
  compile_scalar_novec_output=$(
      {
          $compiler $flags_scalar_novec -o "$new_dump_dir/scalarnovec.o" "$sources_scalar_novec" &&
          ar rvs "$new_dump_dir/libscalarnovec.a" "$new_dump_dir/scalarnovec.o"
      } 2>&1
  )
  compile_scalar_novec_status=$?

  compile_scalar_vec_output=$(
      {
          $compiler $flags_scalar_vec -o "$new_dump_dir/scalarvec.o" "$sources_scalar_vec" &&
          ar rvs "$new_dump_dir/libscalarvec.a" "$new_dump_dir/scalarvec.o"
      } 2>&1
  )
  compile_scalar_vec_status=$?

  compile_vec_output=$(
      {
          $compiler $flags_vec -o "$new_dump_dir/vec.o" "$sources_vec" &&
          ar rvs "$new_dump_dir/libvec.a" "$new_dump_dir/vec.o"
      } 2>&1
  )
  compile_vec_status=$?

  compile_main_output=$($compiler "$sources_main" $flags_main -o "${new_dump_dir}/${benchId}" 2>&1)
  compile_main_status=$?

  # If all statuses are 0
  compile_status=$((compile_main_status + compile_vec_status + compile_scalar_vec_status + compile_scalar_novec_status))

  # Append the new dump dir to the text file dumps directory
  is_best_or_autotune=""
  if [ "$best" = true ]; then
    is_best_or_autotune="best"
  else
    is_best_or_autotune="autotune"
  fi
  echo "${machine}, ${compiler_version}, ${is_best_or_autotune}, ${timestamp}" >> "$dump_dir/benchId${benchId}.txt"

  if [ $compile_status -eq 0 ]; then
    # Run the binary and capture its output
    (cd "$new_dump_dir" && taskset -c 0 "./$benchId") 2>&1
    print_line
    # Print and save compiler output after the binary output
    echo "Compilation succeeded."
    print_line
    echo "compile_scalar_novec_output: "
    echo "$compile_scalar_novec_output"
    print_line
    echo "compile_scalar_vec_output: "
    echo "$compile_scalar_vec_output"
    print_line
    echo "compile_vec_output: "
    echo "$compile_vec_output"
    print_line
    echo "compile_main_output: "
    echo "$compile_main_output"

    echo "$compile_output" >> "$log_file"
  else
    echo "Compilation failed."
    echo "compile_scalar_novec_status: $compile_scalar_novec_status"
    echo "compile_scalar_vec_status: $compile_scalar_vec_status"
    echo "compile_vec_status: $compile_vec_status"
    echo "compile_main_status: $compile_main_status"
    print_line
    echo "compile_scalar_novec_output: "
    echo "$compile_scalar_novec_output"
    print_line
    echo "compile_scalar_vec_output: "
    echo "$compile_scalar_vec_output"
    print_line
    echo "compile_vec_output: "
    echo "$compile_vec_output"
    print_line
    echo "compile_main_output: "
    echo "$compile_main_output"
    exit 1
  fi
} | tee -a "$log_file"
