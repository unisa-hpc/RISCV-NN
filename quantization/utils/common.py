#
# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0
#

import os
import datetime
import pathlib as pl


def prepare_recipe_dir(recipe_name: str, abs_dump_dir: str) -> str:
    # Create the recipe directory
    recipe_dir = pl.Path(abs_dump_dir) / ''.join([recipe_name, "_", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")])
    if not os.path.exists(recipe_dir):
        os.makedirs(recipe_dir)
    return recipe_dir.__str__()
