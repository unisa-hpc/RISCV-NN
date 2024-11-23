#!/bin/bash

compiler=g++

# Dynamically set compiler flags based on compiler type
function set_compiler_flags() {
    local compiler="$1"
    local extra_flags="${2:-}"

    # G++ flags
    if [[ "$compiler" =~ g\+\+ ]]; then
        echo "Using G++ compatible flags."
        flags_main="-O3 -march=rv64imacv -fno-tree-vectorize -fno-tree-slp-vectorize ${new_dump_dir}/libvec.a ${new_dump_dir}/libscalarvec.a ${new_dump_dir}/libscalarnovec.a -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_vec="-c -O3 -march=rv64imacv -fno-tree-vectorize -fno-tree-slp-vectorize -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_scalar_vec="-c -O3 -march=rv64imacv -DAUTOVEC -fopt-info-vec -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_scalar_novec="-c -O3 -march=rv64imacv -fno-tree-vectorize -fno-tree-slp-vectorize -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
    fi

    # Clang++ flags
    if [[ "$compiler" =~ clang\+\+ ]]; then
        echo "Using Clang++ compatible flags."
        flags_main="-O3 -march=rv64imacv -fno-vectorize ${new_dump_dir}/libvec.a ${new_dump_dir}/libscalarvec.a ${new_dump_dir}/libscalarnovec.a -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_vec="-c -O3 -march=rv64imacv -fno-vectorize -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_scalar_vec="-c -O3 -march=rv64imacv -DAUTOVEC -fvectorize -Rpass=loop-vectorize -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_scalar_novec="-c -O3 -march=rv64imacv -fno-vectorize -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
    fi
}

function print_line() {
  echo "############################################"
}

function print_line_long() {
  echo "*"
  echo "*"
  echo "*"
  echo "############################################"
}
# get the path to the symlinked script (not the real file, the symlink's path)

script_dir=$(dirname "$0")
dump_dir="$script_dir/../../dumps"
dump_dir=$(realpath "$dump_dir")
delete_dumps=false

# Process all arguments and handle -d and compiler flags
args=()
for arg in "$@"; do
  if [ "$arg" == "-d" ]; then
    delete_dumps=true
  elif [[ "$arg" =~ ^(g\+\+|clang\+\+) ]]; then
    compiler="$arg"
  else
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

# Check the number of remaining arguments
if [ "${#args[@]}" -gt 1 ]; then
  echo "Usage: $0 [-d] [g++|clang++] [extra_flags]"
  exit 1
fi

extra_flags=${args[0]:-}  # First argument is optional; if not provided, it defaults to an empty string

sources_main="main.cpp"
sources_vec="vectorized.cpp"
sources_scalar_vec="scalar.cpp"
sources_scalar_novec="scalar.cpp"

if [ ! -d "$dump_dir" ]; then
  mkdir "$dump_dir"
fi

timestamp=$(date +"%Y%m%d_%H%M%S")
new_dump_dir="$dump_dir/$timestamp"
mkdir "$new_dump_dir"
log_file="$new_dump_dir/output_log_$timestamp.txt"

# Get abs path of sources_main
abs_main=$(realpath "$script_dir/$sources_main")
# Get the dir name of the folder containing sources_main
echo "sources_main: $sources_main"
echo "abs_main: $abs_main"
benchId=$(basename "$(dirname "$abs_main")")

# Set compiler flags dynamically
set_compiler_flags "$compiler" "$extra_flags"

{
  compiler_version=$($compiler --version | head -n 1)
  echo "Arch: RISCV 64"
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
  echo "$new_dump_dir" >> "$dump_dir/benchId${benchId}.txt"

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
