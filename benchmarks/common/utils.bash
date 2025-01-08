#!/bin/bash

cleanup_directories() {
    local input_file="$1"

    # Check if input file exists
    if [[ ! -f "$input_file" ]]; then
        echo "Error: Input file '$input_file' not found"
        return 1
    fi

    # Process each line
    while IFS=, read -r prefix path; do
        # Trim whitespace from path
        path=$(echo "$path" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

        # Check if directory exists
        if [[ -d "$path" ]]; then
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

