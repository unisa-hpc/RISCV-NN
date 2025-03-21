#!/bin/bash

#
# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0
#

cleanup_directories() {
  local input_file="$1"
  local machine="$2"
  local compiler_exec="$3"

  local compiler_version=$($compiler_exec --version | head -n 1)

  if [[ ! -f "$input_file" ]]; then
    echo "Error: Input file '$input_file' not found"
    return 1
  fi

  if [[ -z "$compiler" ]]; then
    echo "Error: Compiler version must be specified"
    return 1
  fi

  local input_dir=$(dirname "$input_file")
  local temp_file="${input_file}.tmp"

  # Create an empty temp file
  : > "$temp_file"

  while IFS=, read -r val_machine val_compiler_ver val_path; do
    # Skip empty lines
    [[ -z "$val_machine" && -z "$val_compiler_ver" && -z "$val_path" ]] && continue

    # Trim whitespace from all fields
    val_machine=$(echo "$val_machine" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    val_compiler_ver=$(echo "$val_compiler_ver" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    val_path=$(echo "$val_path" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    full_path="$input_dir/$val_path"

    if [[ "$val_machine" == "$machine" && "$val_compiler_ver" == "$compiler_version" ]]; then
      if [[ -d "$full_path" ]]; then
        echo "Deleting directory: $full_path"
        echo "  - Machine: $val_machine"
        echo "  - Compiler: $val_compiler_ver"
        rm -rf "$full_path"
      else
        echo "Warning: Directory not found: $full_path"
        echo "  - Machine: $val_machine"
        echo "  - Compiler: $val_compiler_ver"
      fi
    else
      # Preserve non-matching lines
      echo "$val_machine,$val_compiler_ver,$val_path" >> "$temp_file"
    fi
  done < "$input_file"

  # Avoid overwriting if no lines are written
  if [[ -s "$temp_file" ]]; then
    mv "$temp_file" "$input_file"
  else
    rm -f "$input_file" "$temp_file"
  fi
}

delete_flag_handling() {
  local input_file="$1"
  local local_dump_dir="$2"
  local machine="$3"
  local compiler_exec="$4"

  if [[ -d "$local_dump_dir" ]]; then
    echo "Deleting directories and updating file for machine: $machine"
    cleanup_directories "$input_file" "$machine" "$compiler_exec"

    # Check if the input file is now empty
    if [[ ! -s "$input_file" ]]; then
      echo "Input file is empty. Deleting the autotuner.json as well (if exists)."
      # rm -rf "$local_dump_dir"
      [ -f "$local_dump_dir/autotuner.json" ] && rm -f "$local_dump_dir/autotuner.json"
    fi
  else
    echo "Warning: Local dump directory '$local_dump_dir' not found."
    exit 1
  fi
  exit 0
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

parse_autotuner_best_conf_json() {
    local json_file="$1"
    local benchId="$2"
    local hw="$3"
    local compiler_version="$4"
    local N="$5"

    local json=""

    if ! command -v jq >/dev/null 2>&1; then
        echo "Error: jq is not installed. Please install jq and try again." >&2
        exit 1
    fi

    # Check if the json file exists
    if [ ! -f "$json_file" ]; then
        echo "Error: JSON file '$json_file' not found"
        return 1
    fi

    # Read the JSON file
    json=$(cat "$json_file")

    # Extract the configuration for the specified benchId, hw, compiler, and N
    conf=$(echo "$json" | jq -c ".[\"$benchId\"][\"$hw\"][\"$compiler_version\"][\"$N\"]")

    # Check if the conf exists
    if [ "$conf" == "null" ]; then
        echo "No configuration found for benchId=$benchId, hw=$hw, compiler=$compiler_version, N=$N"
        return 1
    fi

    # Parse the keys and values in conf
    for key in $(echo "$conf" | jq -r 'keys[]'); do
        value=$(echo "$conf" | jq -r --arg k "$key" '.[$k]')

        # Define the variable globally using eval
        eval "$key=$value"
    done
}
