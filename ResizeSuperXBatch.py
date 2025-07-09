import torch
import argparse
import os
import datetime
from ResizeSuperX import ResizeSuperX

class ResizeSuperXBatch():
    def __init__(self, modelFolderPath: str, image_input_folder: str, image_output_folder: str, scaleFactor: int, shouldDenoiseInputImage: bool, sharpness: float|None, devicearg: str|None, gpuIndex: int):
        self.modelFolderPath = modelFolderPath
        self.image_input_folder = image_input_folder
        self.image_output_folder = image_output_folder
        self.scaleFactor = scaleFactor
        self.shouldDenoiseInputImage = shouldDenoiseInputImage
        self.sharpness = sharpness
        self.devicearg = devicearg
        self.gpuIndex = gpuIndex

        device = self.getCudaDeviceName(devicearg, gpuIndex)
        self.device = device

    def run(self):
        start = datetime.datetime.now()

        imageFileNames = self.getListOfFiles(self.image_input_folder)
        index = 1
        for imageFileName in imageFileNames:
            # get file name without extension or directory
            filename = os.path.basename(imageFileName)
            # Use splitext() to get filename and extension separately.
            (justfilename, fileextension) = os.path.splitext(filename)
            outputImageFullPath = os.path.join(self.image_output_folder,f'{justfilename}_{self.scaleFactor}X.png')

            print(f'{index}/{len(imageFileNames)} - converting {filename} to {self.scaleFactor}x')
            index += 1

            resizeSuperX = ResizeSuperX(
                modelFolderPath=self.modelFolderPath,
                image_input=imageFileName,
                image_output=outputImageFullPath,
                scaleFactor=self.scaleFactor,
                shouldDenoiseInputImage=self.shouldDenoiseInputImage,
                sharpness=self.sharpness,
                devicearg=self.devicearg,
                gpuIndex=self.gpuIndex)
            resizeSuperX.run()
        end = datetime.datetime.now()
        duration = (end - start)

        # Extract hours, minutes, and seconds from the duration
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        # Format the duration as h:m:s
        formatted_duration = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

        print(f"took {formatted_duration}")

    def getCudaDeviceName(self, deviceName: str="cuda", gpuNumber: int=0) -> str:
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

    def getListOfFiles(self, dirName: str, traversal: bool = False) -> list[str]:
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            filename = entry
            # If entry is a directory then get the list of files in this directory
            if traversal and os.path.isdir(fullPath):
                allFiles = allFiles + self.getListOfFiles(fullPath)
            else:
                if entry.endswith('.jpg') or entry.endswith('.png'):
                    allFiles.append(fullPath)

        return allFiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Parser for getting arguments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model_folder_path', required=False, default='models', type=str, dest='model_folder_path', help='The directory where the model pth files are located.')
    parser.add_argument('--image_input_folder', required=True, default='images/inputs', type=str, dest='image_input_folder', help='The image folder path of the source images to enhance.')
    parser.add_argument('--image_output_folder', required=True, default='images/outputs', type=str, dest='image_output_folder', help='The image folder path where to save the generated enhanced image file.')
    parser.add_argument('--scale', type=int, default=2, help='scale image: 2, 4, 8, 16, 32. Not supporting 4x or 8x as of now')
    parser.add_argument("--denoise_input_image", required=False, default=False, type=bool, dest="denoise_input_image", help='If true will denoise the input image prior to enlarge it.')
    parser.add_argument('--sharpness', type=float, default=None, help='the sharpness index')
    parser.add_argument('--device', required=False, default=None, type=str, dest='device', help='None, cpu or cuda')
    parser.add_argument('--gpu_index', required=False, default=0, type=int, dest='gpu_index', help='if device is cuda can specify the gpu number if many are available')

    args = parser.parse_args()

    app = ResizeSuperXBatch(args.model_folder_path, args.image_input_folder, args.image_output_folder, args.scale, args.denoise_input_image, args.sharpness, args.device, args.gpu_index)
    app.run()