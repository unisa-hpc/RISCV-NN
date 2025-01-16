import pathlib


def get_all_hw_names(dump_dir: str, bench_id: str) -> [str]:
    # check if f'benchId{bench_id}.txt' exists
    if not pathlib.Path(dump_dir).joinpath(f'benchId{bench_id}.txt').exists():
        print(f'File {dump_dir}/benchId{bench_id}.txt does not exist.')
        return []
    # read all the lines from the file at dump_dir/benchId.txt
    with open(pathlib.Path(dump_dir).joinpath(f'benchId{bench_id}.txt'), 'r') as f:
        lines = f.readlines()

    hw_names = []
    for line in lines:
        # skip if line is empty
        if not line.strip():
            continue
        # find all the json files in current_run_dir
        line = line.strip()
        # seperate hw, path for each line
        hw, line = line.split(',', 1)
        if hw not in hw_names:
            hw_names.append(hw)

    print(f'Found these hardware names: {hw_names}')
    return hw_names


def get_all_json_files(dump_dir: str, bench_id: str, only_this_hw: str) -> [str]:
    """
    Get the abs path of all the json files in the dump directory, recursively.
    """
    if not pathlib.Path(dump_dir).joinpath(f'benchId{bench_id}.txt').exists():
        print(f'File {dump_dir}/benchId{bench_id}.txt does not exist.')
        return []
    # read all the lines from the file at dump_dir/benchId.txt
    with open(pathlib.Path(dump_dir).joinpath(f'benchId{bench_id}.txt'), 'r') as f:
        lines = f.readlines()

    json_files = []
    for line in lines:
        # skip if line is empty
        if not line.strip():
            continue

        # find all the json files in line
        line = line.strip()

        # seperate hw, path for each line
        hw, sub_dump_dir_name = line.split(',', 1)
        sub_dump_dir_name = sub_dump_dir_name.strip()

        if only_this_hw != hw and only_this_hw is not None:
            continue

        sub_dump_dir_path = pathlib.Path(dump_dir) / pathlib.Path(sub_dump_dir_name)
        json_files.extend([str(f) for f in sub_dump_dir_path.rglob('*.json')])

    print(f'Found {len(lines)} sub-dump directories and a '
          f'total of {len(json_files)} json files for benchmark ID {bench_id}')
    return json_files
