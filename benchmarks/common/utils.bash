#!/bin/bash

cleanup_directories() {
  local input_file="$1"
  local machine="$2"

  if [[ ! -f "$input_file" ]]; then
    echo "Error: Input file '$input_file' not found"
    return 1
  fi

  local input_dir=$(dirname "$input_file")
  local temp_file="${input_file}.tmp"

  # Create an empty temp file
  : > "$temp_file"

  while IFS=, read -r prefix path; do
    # Skip empty lines
    [[ -z "$prefix" && -z "$path" ]] && continue

    # Trim whitespace
    prefix=$(echo "$prefix" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    path=$(echo "$path" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    full_path="$input_dir/$path"

    if [[ "$prefix" == "$machine" ]]; then
      if [[ -d "$full_path" ]]; then
        echo "Deleting directory: $full_path"
        rm -rf "$full_path"
      else
        echo "Warning: Directory not found: $full_path"
      fi
    else
      # Preserve non-matching lines
      echo "$prefix,$path" >> "$temp_file"
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

  if [[ -d "$local_dump_dir" ]]; then
    echo "Deleting directories and updating file for machine: $machine"
    cleanup_directories "$input_file" "$machine"

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

parse_autotuner_best_conf_json() {
    local json_file="$1"
    local benchId="$2"
    local hw="$3"
    local N="$4"

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

    # Extract the configuration for the specified benchId, hw, and N
    conf=$(echo "$json" | jq -c ".[\"$benchId\"][\"$hw\"][\"$N\"]")

    # Check if the conf exists
    if [ "$conf" == "null" ]; then
        echo "No configuration found for benchId=$benchId, hw=$hw, N=$N"
        return 1
    fi

    # Parse the keys and values in conf
    for key in $(echo "$conf" | jq -r 'keys[]'); do
        value=$(echo "$conf" | jq -r --arg k "$key" '.[$k]')

        # Define the variable globally using eval
        eval "$key=$value"
    done
}
