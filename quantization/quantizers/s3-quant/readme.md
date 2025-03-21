# S^3 Quantizer
This is the official implementation of the S^3 quantizer, with minor changes to use it in the recipe scheme.

See their git repo [here](https://github.com/huawei-noah/noah-research/tree/master/S3-Training).  
Also see the fork of the author that seems to be more up-to-date [here](https://github.com/xinlinli170/noah-research/tree/master/S3-Training/s3-training-neurips-2021).  

## Modifications
- Added 8-bit quantization support.
- Added a pytorch buffer to store the quantized weights on pytorch.save().