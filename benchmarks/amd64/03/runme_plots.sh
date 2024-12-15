#!/bin/bash

echo "You can disable scalar kernels and verification passes by adding optional argument: -DSKIP_SCALAR_AND_VERIFICATION=1"

optional_flags=""
if [ $# -gt 0 ]; then
  optional_flags="$1"
  echo "Optional flags: $optional_flags"
fi

bash build.amd64.00.sh -d # wipe the dumps dir.

# loop to generate values in list [3, 16, 64, 256, 1024]
input_size_vals=(512 1024)
ch_in_vals=(16 256 1024)
ch_out_vals=(16 256 1024)
kernel_vals=(3 5 7)
stride_vals=(1 2)
unroll0_vals=(1)
unroll1_vals=(1 2)
unroll2_vals=(1 3 5 7)
unroll3_vals=(1 3 5 7)

index=0
total_benchmarks=$(( ${#input_size_vals[@]} * ${#ch_in_vals[@]} * ${#ch_out_vals[@]} * ${#kernel_vals[@]} * ${#stride_vals[@]} * ${#unroll0_vals[@]} * ${#unroll1_vals[@]} * ${#unroll2_vals[@]} * ${#unroll3_vals[@]} ))

# wipe progress txt file
echo "" > /tmp/progressBenchId03.txt

for input_width in "${input_size_vals[@]}"; do
  for input_height in "${input_size_vals[@]}"; do
    for kernel_size in "${kernel_vals[@]}"; do
      for ch_in in "${ch_in_vals[@]}"; do
        for ch_out in "${ch_out_vals[@]}"; do
          for stride in "${stride_vals[@]}"; do
            for u0 in "${unroll0_vals[@]}"; do
              for u1 in "${unroll1_vals[@]}"; do
                for u2 in "${unroll2_vals[@]}"; do
                  for u3 in "${unroll3_vals[@]}"; do
                    index=$((index+1))
                    echo  "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
                    echo "*** benchmark $index out of $total_benchmarks (percent: $((index*100/total_benchmarks))%)"
                    echo "Benchmarking for Input Width of $input_width, Input Height of $input_height, Kernel Size of $kernel_size, Channels In of $ch_in, Channels Out of $ch_out, Stride of $stride, Unroll Factors of $u0, $u1, $u2, $u3."
                    echo  "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
                    bash build.amd64.00.sh "-DI_H=$input_height -DI_W=$input_width -DK_H=$kernel_size -DK_W=$kernel_size -DC_I=$ch_in -DC_O=$ch_out -DS_X=$stride -DS_Y=$stride -DUNROLL_FACTOR0=$u0 -DUNROLL_FACTOR1=$u1 -DUNROLL_FACTOR2=$u2 -DUNROLL_FACTOR3=$u3 ${optional_flags}"
                    # save progress into a text file in /tmp including the percentage of completion and all the parameters with a header line
                    echo "Percent: $((index*100/total_benchmarks))%, Input Width: $input_width, Input Height: $input_height, Kernel Size: $kernel_size, Channels In: $ch_in, Channels Out: $ch_out, Stride: $stride, Unroll Factors: $u0, $u1, $u2, $u3" >> /tmp/progressBenchId03.txt
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

python ../../common/plot.bench_03.py --dumps-dir $(pwd)/../../dumps --out $(pwd)/../../dumps --benchid 03
