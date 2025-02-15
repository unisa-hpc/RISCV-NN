#!/bin/bash
#SBATCH --job-name=run_all_amd64
#SBATCH --output=run_all_amd64.out
#SBATCH --error=run_all_amd64.err
#SBATCH --time=23:00:00
#SBATCH --partition=xxxxxxxxxxxxx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=xxxxxxxxxxxxxxx

# Source the bash script to enable conda and spack
source ~/riscvnn_rootdir/activate_env.sh

# Delete the folder foobar in the home directory
# rm -rf ~/benchmarks/dumps

# Your commands go here
echo "Running my job on $(hostname)"
bash run_all.sh Xeon8260