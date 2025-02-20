#!/bin/bash
set -e

# Usage: $0 <machine> <compiler> [--best] [--extra <extra>] [-d]
#
# Arguments:
#   <machine>: The name of the target machine (required).
#   <compiler>: The compiler to use (required).  Should be "g++" or "clang++" (or variations like g++-11, clang++-14).
#   --best:    A flag indicating that the best optimization options should be used (optional).
#   --extra <extra>: An extra string argument that can be passed to the script (optional).
#   -d:       A flag indicating that some files should be deleted (optional).
#
# Example:
#   ./script.sh mymachine g++ --best --extra "-O3 -Wall" -d


# Dynamically set compiler flags based on compiler type
function set_compiler_flags() {
    local compiler_path="$1"
    local extra_flags="$2"
    local is_gcc_flag="$3"
    local is_clang_flag="$4"
    echo "Compiler passed to set_compiler_flags: $compiler_path"

    # Check is_gcc_flag and is_clang_flag
    if [[ "$is_gcc_flag" -eq 1 ]]; then
        echo "Using G++ compatible flags."
        flags_main="-O3 -march=native -ffast-math -fno-tree-vectorize -fno-tree-slp-vectorize ${new_dump_dir}/libvec.a ${new_dump_dir}/libscalarvec.a ${new_dump_dir}/libscalarnovec.a -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_vec="-c -O3 -march=native -ffast-math -fno-tree-vectorize -fno-tree-slp-vectorize -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_scalar_vec="-c -O3 -march=native -ffast-math -DAUTOVEC -fopt-info-vec-all -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_scalar_novec="-c -O3 -march=native -ffast-math -fno-tree-vectorize -fno-tree-slp-vectorize -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
    elif [[ "$is_clang_flag" -eq 1 ]]; then
        echo "Using Clang++ compatible flags."
        flags_main="-O3 -march=native -ffast-math -fno-vectorize ${new_dump_dir}/libvec.a ${new_dump_dir}/libscalarvec.a ${new_dump_dir}/libscalarnovec.a -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_vec="-c -O3 -march=native -ffast-math -fno-vectorize -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_scalar_vec="-c -O3 -march=native -ffast-math -DAUTOVEC -Rpass=loop-vectorize -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
        flags_scalar_novec="-c -O3 -march=native -ffast-math -fno-vectorize -Wall -Wextra -v -I$script_dir/../../common $extra_flags"
    else
        echo "Error: Unrecognized compiler '$compiler_path'. Must be g++ or clang++ (with optional version number)"
        exit 1
    fi
}

# Parse the command-line arguments and export these variables: machine, compiler, flag_best, flag_delete, extra, is_gcc, is_clang
parse_arguments() {
    machine="$1"
    compiler="$2"
    flag_best=0
    flag_delete=0
    extra=""
    is_gcc=0
    is_clang=0

    # Shift the positional arguments
    shift 2

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --best)
                flag_best=1
                shift
                ;;
            --extra)
                extra="$2"
                shift 2
                ;;
            -d)  # Handle the -d option
                flag_delete=1
                shift
                ;;
            *)
                echo "Invalid argument: $1"
                exit 1
                ;;
        esac
    done

    # Check for GCC variants
    if [[ "$compiler" == *"clang++"* ]]; then
        is_gcc=0
        is_clang=1
    elif [[ "$compiler" == *"g++"* ]]; then
        is_gcc=1
        is_clang=0
    fi

    echo "## Parsed build script arguments:"
    echo "## Machine: $machine"
    echo "## Compiler: $compiler"
    echo "## Flag Best: $flag_best"
    echo "## Flag Del: $flag_delete"
    echo "## Extra: $extra"
    echo "## Is GCC: $is_gcc"
    echo "## Is Clang: $is_clang"
}

# Check for mandatory arguments
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <machine> <compiler> [--best] [--extra <extra>] [-d]"
    exit 1
fi

# Call the function to parse and process arguments
# machine, compiler, flag_best, flag_delete, extra, is_gcc, is_clang
parse_arguments "$@"

# Use the parsed variables


# get the path to the symlinked script (not the real file, the symlink's path)
script_dir=$(dirname "$0")
dump_dir="$script_dir/../../dumps"
dump_dir=$(realpath "$dump_dir")
source "$script_dir/../../common/utils.bash"


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

if [[ "$flag_delete" -eq 1 ]]; then
  echo "Deleting..."
  rm -rf "$new_dump_dir"
  delete_flag_handling "$dump_dir/benchId${benchId}.txt" "$dump_dir" "$machine" "$compiler"
fi

# Set compiler flags dynamically
set_compiler_flags "$compiler" "$extra" "$is_gcc" "$is_clang"

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
  if [[ "$flag_best" -eq 1 ]]; then
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
