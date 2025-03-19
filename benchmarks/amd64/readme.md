# How to

## Install The Dependencies

Make sure you have enough space on your home partition (35GB+).
Make sure that the current working directory of your terminal is `benchmarks/amd64`.

> If you don't have CUDA/Nvidia GPU, you have to delete `pycuda` from `../../scripts/deps.amd64.sh` manually. 
> Otherwise, the script will fail.

```bash
bash ../../scripts/deps.amd64.sh
```

## Setup the Environment

```bash
source ~/riscvnn_rootdir/activate_env.sh
```

## Run The Benchmarks

Now, run the benchmarks using the following command:

```bash
bash run.sh <NameOfYourCpuWithoutSpaces>
```

if the execution has failed, please delete `benchmarks/dumps` directory first and then try again.

## Stash the Results

Copy the `dumps` directory somewhere safe and don't mix it with other dumps. Later you have to feed its path as a
`--dumps=` argument to the Python plotting script.

