# RISCV-NN
Exploring RISCV for fast NN inference.

## Workflow
01. Set the weights and activations data type configs.
02. Select the model:
    - ResNet18, CIFAR10
    - Custom, MNIST (FC/Conv2)
03. Export the full model to a .onnx file.
04. Add the config to the .onnx file.
05. Add the signs and shifts to the .onnx file.
06. Run IREE on the full model ONNX file to generate RVV enabled binary for RISCV.
07. Given C++ templates, tailor the kernels to each layer in the ONNX file.
08. Add the needed CMake scripts.
09. Build the project on RISCV.
10. Run the inference on the RISCV.
11. Compare the runtime results against IREE.
12. 

## TODO
1. Decide on multithreading. How does IREE handle it?
2. 
