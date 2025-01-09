#!/bin/bash

cleanup_directories() {
  local input_file="$1"

  # Check if input file exists
  if [[ ! -f "$input_file" ]]; then
    echo "Error: Input file '$input_file' not found"
    return 1
  fi

  # Get the path containing the input file
  local input_dir=$(dirname "$input_file")

  # Process each line
  while IFS=, read -r prefix path; do
    # Trim whitespace from path
    path=$(echo "$path" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    path="$input_dir/$path"

    # Check if directory exists
    if [[ -d "${path}" ]]; then
      echo "Deleting directory: $path"
      rm -rf "$path"
    else
      echo "Warning: Directory not found: $path"
    fi
  done < "$input_file"

  # delete the input file
  rm -f "$input_file"
}

delete_flag_handling() {
  local input_file="$1" # the txt file containing lines: hw, path of new_dump_dir
  local local_dump_dir="$2"

  if [ "$delete_dumps" = true ] && [ -d "$dump_dir" ]; then
    echo "Deleting only the directories listed in $input_file"
    cleanup_directories "$input_file"
    echo "Deleted all contents of directories listed in $input_file , exiting."
    # delete the local_dump_dir
    rm -rf "$local_dump_dir"
    exit 0
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

function setup_autotuner_args() {
  # Global flags, not local
  flag_delete_dumps=false
  flag_auto_tune=false
  machine=""
  args=""

  # Process the arguments and set the flags and the machine name. Add the rest of the arguments to the args string
  for i in "$@"; do
    case $i in
      -d)
          flag_delete_dumps=true
          ;;
      --auto-tune)
          flag_auto_tune=true
          ;;
      --machine=*)
          machine="${i#*=}"
          ;;
      *)
          args+="$i "
          ;;
    esac
  done

  # Make sure -d and --auto-tune flags are not set at the same time
  if [ "$flag_delete_dumps" = true ] && [ "$flag_auto_tune" = true ]; then
    echo "Error: -d and --auto-tune flags cannot be set at the same time."
    exit 1
  fi

  # Make sure the machine name is provided
  if [ -z "$machine" ]; then
    echo "Error: --machine argument is mandatory"
    echo "Usage: $0 --machine=<string> [-d] [--auto-tune] [g++|clang++] [extra_flags]"
    exit 1
  fi
}