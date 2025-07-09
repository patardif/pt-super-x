import torch
from torch import nn
import numpy as np
from PIL import Image
import cv2
from utils.TileMax import TileMax
from utils.BaseResizeSuperX import BaseResizeSuperX

def sharpen_image(pil_image: Image, sharpen_amount: float):
    '''
    Apply sharpening to a PIL image.
    '''
    # Load the image
    image = np.array(pil_image)

    # Apply the sharpening kernel
    sharpened = sharpen_image_tensor(image, kernel_size=(5, 5), amount=sharpen_amount)

    img = Image.fromarray(sharpened)

    return img

def sharpen_image_tensor(image_tensor: np.array, kernel_size: tuple[int ,int ] =(5, 5), amount: float =1.0) -> np.array:
    '''
    Apply sharpening to an image Tensor.
    '''
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(image_tensor, kernel_size, 0)
    # Sharpen the image
    sharpened = cv2.addWeighted(image_tensor, 1 + amount, blurred, -amount, 0)
    return sharpened

def denoise_smooth_image(pil_image: Image, diameter_each_pixel_neighborhood: int, sigma_color: int, sigma_space: int):
    '''
    d: Diameter of each pixel neighborhood.
    sigmaColor: Value of σ  in the color space. The greater the value, the colors farther to each other will start to get mixed.
    sigmaSpace: Value of  σ  in the coordinate space. The greater its value, the more further pixels will mix together, given that their colors lie within the sigmaColor range.
    '''
    numpy_image = np.array(pil_image)
    numpy_image_filtered = cv2.bilateralFilter(numpy_image, d=diameter_each_pixel_neighborhood, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return Image.fromarray(numpy_image_filtered)

def generate_image(baseResizeSuperX: BaseResizeSuperX, scale: int, model: nn.Module, fn: str, tile: Image=None) -> Image:
    '''
    Take a tile image as input and apply the super-resolution model to it.
    Return the enhanced tile image as PIL image.
    '''
    # Load images
    if tile is None:
        tile = Image.open(fn)
    img = tile
    img = img.convert('RGB')

    enhanced_image = baseResizeSuperX.apply_model(model, img)

    return enhanced_image

def get_gpu_memory(gpu_index: int):
    '''
    Check how much GPU ram is available.
    '''
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            if gpu_index == i:
                free_mem = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                total_mem = torch.cuda.get_device_properties(i).total_memory
                used_mem = torch.cuda.memory_allocated(i)
                return total_mem

    else:
        print("No GPU available")

def getPerformanceTileMax(device: str, gpu_ram: int | None) -> TileMax:
    '''
    Provide working tested values of tile's size based on the GPU memory available, for trained 2x, 16x and 32x.
    '''
    if device != 'cpu':
        # code here has been tested for 24GB or 12GB GPU ram
        if gpu_ram <= 12 * 1024**3: # 12  GB
            return TileMax(max_16=128, max_8=256, max_2=1024)
        elif gpu_ram <= 24 * 1024**3: # 24GB
            return TileMax(max_16=256, max_8=384, max_2=1536)
        elif gpu_ram > 24 * 1024**3: # 24GB+
            return TileMax(max_16=512, max_8=512, max_2=1536)

    return TileMax(max_16=256, max_8=384, max_2=1536) # default for cpu and any unknown

def getCudaDeviceName(deviceName: str ="cuda", gpuNumber: int =0):
    result = "cuda"
    if deviceName is not None and deviceName == "cuda":
        # return cuda gpu number if available otherwise return default cuda or cpu if no cuda available
        if torch.cuda.is_available():
            if gpuNumber is not None:
                num_of_gpus = torch.cuda.device_count()
                if num_of_gpus > gpuNumber:
                    result = f"cuda:{gpuNumber}"
        else:
            result = "cpu"
    else:
        result = "cpu"
    return result

def open_PIL_image_as_RGB(image_path: str, existing_pil_image: Image=None):
    if existing_pil_image is None:
        image = Image.open(image_path)
    else:
        image = existing_pil_image

    return image.convert('RGB')