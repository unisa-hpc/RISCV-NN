import pathlib


def get_all_tuples(dump_dir: str, bench_id: int) -> [str]:
    # check if bench_id is a valid integer
    if not isinstance(bench_id, int):
        raise ValueError(f'bench_id should be an integer, got {bench_id}')

    # get the two digit string representation of bench_id with zeros padded to the left
    if bench_id >99:
        raise ValueError(f'bench_id should be less than 100, got {bench_id}')
    bench_id_str = str(bench_id).zfill(2)

    # check if f'benchId{bench_id}.txt' exists
    if not pathlib.Path(dump_dir).joinpath(f'benchId{bench_id_str}.txt').exists():
        raise ValueError(f'File {dump_dir}/benchId{bench_id_str}.txt does not exist.')
    # read all the lines from the file at dump_dir/benchId.txt
    with open(pathlib.Path(dump_dir).joinpath(f'benchId{bench_id_str}.txt'), 'r') as f:
        lines = f.readlines()
    tuples = []
    for line in lines:
        # skip if line is empty
        if not line.strip():
            continue
        # find all the json files in current_run_dir
        line = line.strip()
        # seperate hw, path for each line
        t = line.split(',', 2)
        # strip all elements in the tuple
        t_stripped = [x.strip() for x in t]
        if len(t_stripped) != 3:
            raise ValueError('Invalid line: ' + line)
        tuples.append(t_stripped)
    return tuples


def get_all_hw_names(dump_dir: str, bench_id: int) -> [str]:
    hw_names = []
    all_tuples = get_all_tuples(dump_dir, bench_id)
    for entry in all_tuples:
        if entry[0] not in hw_names:
            hw_names.append(entry[0])
    print(f'Found these hardware names: {hw_names}')
    return hw_names


def get_all_compiler_names(dump_dir: str, bench_id: int) -> [str]:
    compiler_names = []
    all_tuples = get_all_tuples(dump_dir, bench_id)
    for entry in all_tuples:
        if entry[1] not in compiler_names:
            compiler_names.append(entry[1])
    print(f'Found these compiler names: {compiler_names}')
    return compiler_names


def get_all_json_files(dump_dir: str, bench_id: int, only_this_hw: str, only_this_compiler: str) -> [str]:
    """
    Get the abs path of all the json files in the dump directory, recursively.
    """
    all_tuples = get_all_tuples(dump_dir, bench_id)
    json_files = []
    for hw, compiler, sub_dump_dir_name in all_tuples:
        if only_this_hw != hw and only_this_hw is not None:
            continue
        if only_this_compiler != compiler and only_this_compiler is not None:
            continue
        sub_dump_dir_path = pathlib.Path(dump_dir) / pathlib.Path(sub_dump_dir_name)
        json_files.extend([str(f) for f in sub_dump_dir_path.rglob('*.json')])

    print(f'Found {len(all_tuples)} sub-dump directories and a '
          f'total of {len(json_files)} json files for benchmark ID {bench_id}')
    return json_files

