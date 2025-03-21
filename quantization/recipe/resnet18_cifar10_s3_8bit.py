#
# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0
#

import utils.common as cmn
import pathlib as pl
import subprocess


def run_recipe(abs_root_dir: str, abs_dump_dir: str, abs_data_dir: str):
    recipe_name = 'resnet18_cifar10'
    abs_recipe_dir = pl.Path(cmn.prepare_recipe_dir(recipe_name, abs_dump_dir))
    raw_bash = f"""
#!/bin/bash
echo "Recipe: {recipe_name}"
echo ">> Root directory is {abs_root_dir}"
echo ">> Dump directory is {abs_dump_dir}"
echo ">> Data directory is {abs_data_dir}"
echo ">> Recipe directory is {abs_recipe_dir}"
echo "Running the recipe..."
python {abs_root_dir}/quantizers/s3-quant/main.py -b=512 --cifar10 --dump={abs_recipe_dir} --data={abs_data_dir}
    """
    with open(abs_recipe_dir/'recipe.sh', 'w') as file:
        file.write(raw_bash)

    print(f"Please run the following command in the terminal:\n")
    print(f"cd {abs_recipe_dir} && bash recipe.sh")
