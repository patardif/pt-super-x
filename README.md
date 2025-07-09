# Super-Resolution using Residual Dense Blocks
[Patrick Tardif](https://www.patricktardif.com)

The code provided was tested using NVidia RTX-3090 24GB with CUDA. The code can run using CPU but for performance reason, it's highly recommended to use CUDA. For example, enhancing the bird.png image by 8x on cpu took 159 seconds and using CUDA (RTX-3090) took 3s, a factor of 53. The point of this project is to be able to run it using cheap GPUs such as the RTX-3080 or RTX-3090 or better.

## Requirements
A list of python modules used which include a way to install pytorch using CUDA. 
```shell script
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy==2.2.3
pip install pillow==11.1.0
pip install opencv-python==4.11.0.86
```
# Pretrained Models
The models, 2x, 8x are available to download at Hugging Face (https://huggingface.co/patardi/pt-super-x).

Add the downloaded model pth files in the folder called "models".

# Testing
To test the super-resolution and output 2x, 8x images, use the test.py as example. If you need 32x, 64x, 128x, 256x output images, please go to my personal website and contact me from there.



