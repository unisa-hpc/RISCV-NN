from matplotlib import pyplot as plt
import argparse
import json
import pathlib


def get_all_json_files(dump_dir) -> [str]:
    """
    Get the abs path of all the json files in the dump directory, recursively.
    """
    return list(pathlib.Path(dump_dir).rglob('*.json'))


def parse_json(abs_path: str) -> dict:
    """
    Parse the json file and return the data.
    """
    with open(abs_path) as f:
        return json.load(f)


if __name__ == '__main__':
    # add -d argument for dumps dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps-dir', type=str, required=True)
    args = parser.parse_args()

    all_jsons = get_all_json_files(args.dumps_dir)
    parsed_runs = [parse_json(j) for j in all_jsons]
    grouped_runs_by = {} # name -> unroll_factor -> [runtimes]
    for run in parsed_runs:
        name = run['name']
        if name not in grouped_runs_by:
            grouped_runs_by[name] = {}
        unroll_factor = run['pairs']['unroll_factor']

        grouped_runs_by[name][unroll_factor] = run['median']

    # for each name, plot different unroll factors, plot them on top of each other in the same figure
    f = plt.figure()
    all_x = []
    for name, unroll_factors in grouped_runs_by.items():
        x = sorted(unroll_factors.keys())
        all_x.extend(x)
        y = [unroll_factors[uf] for uf in x]
        plt.plot(x, y, marker='o', label=name)
    plt.legend()
    plt.xlabel('Unroll Factor')
    plt.ylabel('Runtime (ms)')
    plt.title('Runtime vs Unroll Factor')
    f.set_size_inches(6.4, 4.8)
    plt.xticks(all_x)
    plt.savefig(pathlib.Path(args.dumps_dir).joinpath('runtime_vs_unroll_factor.png'))
