iree-import-onnx ../data/models/tinyonnx3.onnx -o /tmp/iree.test.mlir
iree-compile /tmp/iree.test.mlir --iree-hal-target-backends=llvm-cpu -o /tmp/iree.test.vmfb
iree-run-module --module=/tmp/iree.test.vmfb --device=local-task --input="1x3x32x32xf32=1"
