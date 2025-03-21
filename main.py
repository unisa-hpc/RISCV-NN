#
# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import os


def verify_dirs(dump_dir, data_dir):
    # Check if dump directory exists
    if not os.path.exists(dump_dir):
        print(f"Dump directory {dump_dir} does not exist.")
        print(f"Creating dump directory {dump_dir}")
        os.makedirs(dump_dir)

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        print(f"Creating dump directory {data_dir}")
        os.makedirs(data_dir)


if __name__ == '__main__':
    # add args for dump and data dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=str, required=False, default='resnet18_cifar10_denseshift_8bits',
                        help='The name of the chosen recipe file without the extension.')
    parser.add_argument('--dump', type=str, required=False, default='dump')
    parser.add_argument('--data', type=str, required=False, default='data')
    args = parser.parse_args()
    verify_dirs(args.dump, args.data)

    # Load the recipe file
    recipe_file = f"recipe/{args.recipe}.py"
    if not os.path.exists(recipe_file):
        print(f"Recipe file {recipe_file} does not exist.")
        exit(1)

    # import the recipe file
    exec(open(recipe_file).read())
    # send the root dir of the project to the recipe
    root = os.path.dirname(os.path.abspath(__file__))
    # convert relative paths to absolute paths if they are not absolute
    args.dump = os.path.abspath(args.dump)
    args.data = os.path.abspath(args.data)

    run_recipe(root, args.dump, args.data)


