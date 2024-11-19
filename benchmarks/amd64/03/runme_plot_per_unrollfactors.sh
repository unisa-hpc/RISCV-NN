bash build.amd64.00.sh -d # wipe the dumps dir.

# loop to generate values in list [3, 16, 64, 256, 1024]
input_size_vals=(64 256 1024)
ch_in_vals=(3 16 64)
ch_out_vals=(16 256)
unroll0_vals=(1 2)
unroll1_vals=(1 2 3 4)
unroll2_vals=(1 2 3)
unroll3_vals=(1 2 3)

for input_width in "${input_size_vals[@]}"; do
  for input_height in "${input_size_vals[@]}"; do
    for ((kernel_size=3; kernel_size<=7; kernel_size+=2)); do
      for ch_in in "${ch_in_vals[@]}"; do
        for ch_out in "${ch_out_vals[@]}"; do
          for stride in 1 2; do
            for u0 in "${unroll0_vals[@]}"; do
              for u1 in "${unroll1_vals[@]}"; do
                for u2 in "${unroll2_vals[@]}"; do
                  for u3 in "${unroll3_vals[@]}"; do
                    echo "Benchmarking for Input Width of $input_width, Input Height of $input_height, Kernel Size of $kernel_size, Channels In of $ch_in, Channels Out of $ch_out, Stride of $stride, Unroll Factors of $u0, $u1, $u2, $u3."
                    bash build.amd64.00.sh "-DI_H=$input_height -DI_W=$input_width -DK_H=$kernel_size -DK_W=$kernel_size -DC_I=$ch_in -DC_O=$ch_out -DS_X=$stride -DS_Y=$stride -DUNROLL_FACTOR0=$u0 -DUNROLL_FACTOR1=$u1 -DUNROLL_FACTOR2=$u2 -DUNROLL_FACTOR3=$u3"
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

#python ../../common/plot.bench_01_02.py --dumps-dir ../../dumps --out "BenchId01.png"