#!/bin/bash
#SBATCH --job-name=xxxxxx
#SBATCH --output=orep.txt
#SBATCH --error=erep.txt
#SBATCH --time=23:00:00
#SBATCH --partition=xxxxxxxxxxxxx
#SBATCH --ntasks=1                    # So we reserve a whole cpu socket with 24 cores
#SBATCH --nodes=1                     # and run our single thread benchmarks on core 0
#SBATCH --sockets-per-node=1          # with taskset -c 0 ....
#SBATCH --cores-per-socket=24         # cat /proc/self/status | grep Cpus_allowed_list
#SBATCH --cpus-per-task=24
#SBATCH --account=xxxxxxxxxxxxxxx

cat /proc/self/status | grep Cpus_allowed_list

# Source the bash script to enable conda and spack
source ~/riscvnn_rootdir/activate_env.sh

# Delete the folder foobar in the home directory
# rm -rf ~/benchmarks/dumps

# Your commands go here
echo "Running my job on $(hostname)"
bash run_all.sh Xeon8260