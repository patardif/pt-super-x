import math
import gc
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import argparse
from PIL import Image
import os
import datetime
from datetime import datetime
import utils.SuperXUtilities as SuperXUtilities
import PT_RRDB_Net as net
from utils.BaseResizeSuperX import BaseResizeSuperX
from utils.TileMax import TileMax
from utils.TileImageWithPadding import TileImageWithPadding
from utils.TileImageWithPaddingArray import TileImageWithPaddingArray

# padding added to an image before doing super x since it has artifacts at edges
PADDING_EDGE_TO_FIX_ARTIFACTS = 45

class ResizeSuperX(BaseResizeSuperX):
    def __init__(self, modelFolderPath: str, image_input: str, image_output: str, scaleFactor: int, shouldDenoiseInputImage: bool, sharpness: float|None, devicearg: str|None, gpuIndex: int):
        self.modelFolderPath = modelFolderPath
        self.image_input = image_input
        self.image_output = image_output
        self.shouldDenoiseInputImage = shouldDenoiseInputImage
        self.scaleFactor = scaleFactor
        self.sharpness = sharpness
        self.devicearg = devicearg
        self.gpuIndex = gpuIndex

        device = SuperXUtilities.getCudaDeviceName(devicearg, gpuIndex)
        self.device = device

        self.GPU_ram = None
        if self.device != 'cpu':
            self.GPU_ram = SuperXUtilities.get_gpu_memory(gpuIndex)

    def run(self):
        '''
        The main external method to call to turn an input image into an enhance/enlarge image.
        '''
        # we do super X and we do not want memory limitation
        Image.MAX_IMAGE_PIXELS = None

        tile_max_for_best_performance = SuperXUtilities.getPerformanceTileMax(self.device, self.GPU_ram)
        self.processStepEnlargeX(self.device, self.image_input, self.image_output, self.modelFolderPath, scaleToApply=self.scaleFactor, tile_max=tile_max_for_best_performance, sharpening_threshold=self.sharpness)

    def processStepEnlargeX(self, device: str, imageInputPath: str, imageOutput: str, modelFolderPath: str, scaleToApply:int=2, tile_max:TileMax=None, sharpening_threshold:float=None):
        '''
        The main internal method processing an image input and turning it into an enhance/enlarge image using the scale provided.
        '''
        with torch.no_grad():

            # default to tile max of 16x
            applyTileMax: int = tile_max.max_16
            apply_sharpening_threshold_first_stage = sharpening_threshold

            if scaleToApply == 2:
                # prepare 2x model
                model = self.getLoadedModelWithStateDict(modelFolderPath, device, 2)
                model.eval()
                applyTileMax = tile_max.max_2

            elif scaleToApply == 8:
                # prepare 2x model
                model = self.getLoadedModelWithStateDict(modelFolderPath, device, 8)
                model.eval()
                applyTileMax = tile_max.max_8

            elif scaleToApply == 16:
                model = self.getLoadedModelWithStateDict(modelFolderPath, device, 16)
                model.eval()
                applyTileMax = tile_max.max_16

            elif scaleToApply == 32:
                model_2x = model = self.getLoadedModelWithStateDict(modelFolderPath, device, 2)
                model = self.getLoadedModelWithStateDict(modelFolderPath, device, 16)
                model.eval()
                model_2x.eval()
                apply_sharpening_threshold_first_stage = None
                applyTileMax = tile_max.max_16

            start_time = datetime.now()
            file_name, file_ext = os.path.splitext(os.path.basename(imageInputPath))

            print(f'enhancing {file_name}{file_ext} to {scaleToApply}X...')

            apply_scale = scaleToApply
            # just to get the original size
            image = SuperXUtilities.open_PIL_image_as_RGB(imageInputPath)

            # apply denoising and smoothing to image before scaling it
            if self.shouldDenoiseInputImage:
                image = SuperXUtilities.denoise_smooth_image(pil_image=image, diameter_each_pixel_neighborhood=5, sigma_color=32, sigma_space=32)

            original_image_width, original_image_height = image.size
            upscaled_image_width = original_image_width * scaleToApply
            upscaled_image_height = original_image_height * scaleToApply

            if scaleToApply == 32:
                apply_scale = 16  # we split that into 2 processes, first at 16x and second at 2x

            tile_max_padded = applyTileMax - PADDING_EDGE_TO_FIX_ARTIFACTS * 2  # need to reduce tile size because we are adding xx padding to resolve artifacts at edges at want GPU Ram < Ram available

            reconstructed_image = self.superX_generate_image_pad(image_file=None, pil_image=image, model=model, scale=apply_scale, tile_max=tile_max_padded, sharpening_threshold=apply_sharpening_threshold_first_stage)

            # for 32x for reprocess the reconstructed image and apply 2x to it
            if scaleToApply == 32:
                reconstructed_image = self.superX_generate_image_pad(image_file=None, pil_image=reconstructed_image, model=model_2x, scale=2, tile_max=tile_max.max_2, sharpening_threshold=sharpening_threshold)

            # Save to file
            reconstructed_image.save(imageOutput)

            del reconstructed_image

            end = datetime.now()
            duration = (end - start_time)

            # Convert timedelta to total milliseconds
            total_milliseconds = duration.total_seconds() * 1000

            # Use divmod to calculate hours and remaining milliseconds
            hours, remaining_milliseconds = divmod(total_milliseconds, 3600000)
            minutes, remaining_milliseconds = divmod(remaining_milliseconds, 60000)
            seconds, remaining_milliseconds = divmod(remaining_milliseconds, 1000)

            # Format the duration as h:m:s:ms
            formatted_duration = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}:{int(remaining_milliseconds):03}"

            print(f'completed  {file_name}{file_ext} from {original_image_width}x{original_image_height} to {upscaled_image_width}x{upscaled_image_height}, took {formatted_duration}')

    def getLoadedModelWithStateDict(self, modelFolderPath: str, device: str, scale:int=2) -> nn.Module:
        model_2x_name = 'xtremertx_model_2x.pth'
        model_8x_name = 'xtremertx_model_8x.pth'
        if scale == 2:
            model_2x = net.PT_RRDB_Net_2x(in_channels=3, out_channels=3, channels=64, num_rrdb=4, growth_channels=32).to(device)
            lm = torch.load(os.path.join(modelFolderPath, model_2x_name), weights_only=False)
            model_2x.load_state_dict(lm.state_dict(), strict=True)
            return model_2x
        if scale == 8:
            model_8x = net.PT_RRDB_Net_8x(in_channels=3, out_channels=3, channels=64, num_rrdb=23, growth_channels=32).to(device)
            lm = torch.load(os.path.join(modelFolderPath, model_8x_name), weights_only=False)
            model_8x.load_state_dict(lm.state_dict(), strict=True)
            return model_8x

        raise NotImplementedError(f'scale {scale} is not supported as of now')

    def superX_generate_image_pad(self, image_file: str|None, pil_image: Image, model: nn.Module, scale: int=2, apply_tiles: bool=True, tile_max: int=256, sharpening_threshold:float=None) -> Image:
        '''
        Will break image into tiles and then for each tile, will enhance them using the super-resolution model.
        Will return an image all stitched together.
        '''
        if apply_tiles:
            tiles, original_size = self.break_into_tiles_pad(image_file, pil_image, tile_max, PADDING_EDGE_TO_FIX_ARTIFACTS)

            new_tiles_TileImageWithPaddingArray = TileImageWithPaddingArray(original_size[0], original_size[1])

            for tile_TileImageWithPadding in tiles:
                new_tile = self.apply_model(model, tile_TileImageWithPadding.getImageWithPadding())
                if sharpening_threshold is not None:
                    new_tile = SuperXUtilities.sharpen_image(new_tile, sharpening_threshold)
                tile_TileImageWithPadding.setSuperResolutionImageWithPadding(new_tile, scale)
                new_tiles_TileImageWithPaddingArray.append(tile_TileImageWithPadding)

            reconstructed_image = new_tiles_TileImageWithPaddingArray.getSuperImage(scale)
        else:
            reconstructed_image = SuperXUtilities.generate_image(self, scale, model, image_file, pil_image)
            if sharpening_threshold is not None:
                reconstructed_image = SuperXUtilities.sharpen_image(reconstructed_image, sharpening_threshold)

        return reconstructed_image

    '''
    Example of tiles' formations
    
    A   B   C   D
    E   F   G   H
    I   J   K   L
    M   N   O   P

    Tile-Row1-Col1:     Tile-Row1-Col2:     Tile-Row1-Col3:     Tile-Row1-Col4:
        A*  B           A   B*  C           B   C*  D           C   D*
        E   F           E   F   G           F   G   H           G   H

    Tile-Row2-Col1:     Tile-Row2-Col2:     Tile-Row2-Col3:     Tile-Row2-Col4:
        A   B           A   B   C           B   C   D           C   D
        E*  F           E   F*  G           F   G*  H           G   H*
        I   J           I   J   K           J   K   L           K   L

    Tile-Row3-Col1:     Tile-Row3-Col2:     Tile-Row3-Col3:     Tile-Row3-Col4:
        E   F           E   F   G           F   G   H           G   H
        I*  J           I   J*  K           J   K*  L           K   L*
        M   N           M   N   O           N   O   P           O   P

    Tile-Row4-Col1:     Tile-Row4-Col2:     Tile-Row4-Col3:     Tile-Row4-Col4:
        I   J           I   J   K           J   K   L           K   L
        M*  N           M   N*  O           N   O*  P           O   P*
    '''
    def break_into_tiles_pad(self, image_path: str, img: Image, tile_size: int, padding: int):
        '''
        Breaks image into tiles using a target section of the image and it's neighbor.
        '''

        image = SuperXUtilities.open_PIL_image_as_RGB(image_path, img)

        image_width, image_height = image.size

        # Calculate the number of tiles
        tiles_per_column = math.ceil(math.ceil(image_width / tile_size))
        tiles_per_row = math.ceil(math.ceil(image_height / tile_size))

        # Create a list to hold the tiles
        tiles = []

        for row in range(tiles_per_row):
            for col in range(tiles_per_column):
                isFirstRow = row == 0
                isFirstCol = col == 0
                isLastRow = row == tiles_per_row - 1
                isLastCol = col == tiles_per_column - 1

                left_inclusive = col * tile_size - padding
                right_exclusive = col * tile_size + tile_size + padding
                upper_inclusive = row * tile_size - padding
                lower_exclusive = row * tile_size + tile_size + padding

                if isFirstRow:
                    upper_inclusive = 0
                    lower_exclusive = tile_size + padding
                    if lower_exclusive > image_height:
                        lower_exclusive = image_height

                if isFirstCol:
                    left_inclusive = 0
                    right_exclusive = tile_size + padding
                    if right_exclusive > image_width:
                        right_exclusive = image_width

                if isLastRow and not isFirstRow:
                    lower_exclusive = image_height
                    upper_inclusive = lower_exclusive - tile_size - padding

                if isLastCol and not isFirstCol:
                    right_exclusive = image_width
                    left_inclusive = right_exclusive - tile_size - padding

                # Crop the tile from the image
                if tiles_per_column == 1 and tiles_per_row == 1:
                    tile = image
                    tiles.append(TileImageWithPadding(tile, image_width, image_height, row, col, isLastRow, isLastCol, 0))
                else:
                    tile = image.crop((left_inclusive, upper_inclusive, right_exclusive, lower_exclusive))
                    tiles.append(TileImageWithPadding(tile, tile_size, tile_size, row, col, isLastRow, isLastCol, padding))

        return tiles, (image_width, image_height)

    def apply_model(self, model: nn.Module, pil_image_rgb: Image) -> Image:
        '''
        Apply the model to the input PIL image and return the enhanced/enlarge image as a PIL Image.
        Also make sure the process is optimized by clearing the cuda cache to lower memory consumption.
        '''
        img = np.asarray(pil_image_rgb).astype(np.float32) / 255.0  # HxWxC
        img = np.transpose(img, [2, 0, 1])  # CxHxW
        img = img[np.newaxis, ...]  # BxCxHxW
        batch_input = Variable(torch.from_numpy(img)).to(self.device)

        batch_output = model(batch_input)
        batch_output = (batch_output).cpu().data.numpy()

        batch_output = np.clip(batch_output[0], 0., 1.)
        batch_output = np.transpose(batch_output, [1, 2, 0])

        enhanced_image = Image.fromarray(np.around(batch_output * 255).astype(np.uint8))

        # here, it's important to clear CUDA if used to free GPU memory and keep memory below thresholds
        del batch_input
        del batch_output
        gc.collect()
        if self.device != 'cpu':
            torch.cuda.empty_cache()

        return enhanced_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Parser for getting arguments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model_folder_path', required=False, default='models', type=str, dest='model_folder_path', help='The directory where the model pth files are located.')
    parser.add_argument('--image_input', required=True, default='images/test.jpg', type=str, dest='image_input', help='The image file path of the source image to enhance.')
    parser.add_argument('--image_output', required=True, default='images/test-4X.jpg', type=str, dest='image_output', help='The image file path of the generated enhanced image file name to save.')
    parser.add_argument('--scale', type=int, default=2, help='scale image: 2, 16, 32. Not supporting 4x or 8x as of now')
    parser.add_argument("--denoise_input_image", required=False, default=False, type=bool, dest="denoise_input_image", help='If true will denoise the input image prior to enlarge it.')
    parser.add_argument('--sharpness', type=float, default=None, help='the sharpness index')
    parser.add_argument('--device', required=False, default=None, type=str, dest='device', help='None, cpu or cuda')
    parser.add_argument('--gpu_index', required=False, default=0, type=int, dest='gpu_index', help='if device is cuda can specify the gpu number if many are available')

    args = parser.parse_args()

    app = ResizeSuperX(args.model_folder_path, args.image_input, args.image_output, args.scale, args.denoise_input_image, args.sharpness, args.device, args.gpu_index)
    app.run()
