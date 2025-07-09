from ResizeSuperX import ResizeSuperX
from ResizeSuperXBatch import ResizeSuperXBatch

if __name__=="__main__":
    resizeSuperX = ResizeSuperX(
        modelFolderPath='models',
        image_input='./test/inputs/bird.png',
        image_output='./test/outputs/bird_8x_cpu.png',
        scaleFactor=8,
        shouldDenoiseInputImage=True,
        sharpness=None,
        devicearg='cpu',
        gpuIndex=0)
    resizeSuperX.run()

    resizeSuperX = ResizeSuperX(
        modelFolderPath='models',
        image_input='./test/inputs/bird.png',
        image_output='./test/outputs/bird_2x.png',
        scaleFactor=2,
        shouldDenoiseInputImage=True,
        sharpness=None,
        devicearg='cuda',
        gpuIndex=0)
    resizeSuperX.run()

    resizeSuperX = ResizeSuperX(
        modelFolderPath='models',
        image_input='./test/inputs/bird.png',
        image_output='./test/outputs/bird_2x_with_sharpening.png',
        scaleFactor=2,
        shouldDenoiseInputImage=True,
        sharpness=1.5,
        devicearg='cuda',
        gpuIndex=0)
    resizeSuperX.run()

    resizeSuperX = ResizeSuperX(
        modelFolderPath='models',
        image_input='./test/inputs/bird.png',
        image_output='./test/outputs/bird_8x.png',
        scaleFactor=8,
        shouldDenoiseInputImage=True,
        sharpness=None,
        devicearg='cuda',
        gpuIndex=0)
    resizeSuperX.run()

    resizeSuperX = ResizeSuperX(
        modelFolderPath='models',
        image_input='./test/inputs/bird.png',
        image_output='./test/outputs/bird_8x_NotDenoised.png',
        scaleFactor=8,
        shouldDenoiseInputImage=False,
        sharpness=None,
        devicearg='cuda',
        gpuIndex=0)
    resizeSuperX.run()

    resizeSuperX = ResizeSuperX(
        modelFolderPath='models',
        image_input='./test/inputs/bird.png',
        image_output='./test/outputs/bird_8x_with_sharpening.png',
        scaleFactor=8,
        shouldDenoiseInputImage=True,
        sharpness=1.5,
        devicearg='cuda',
        gpuIndex=0)
    resizeSuperX.run()

    resizeSuperXBatch = ResizeSuperXBatch(
        modelFolderPath='models',
        image_input_folder='./test/inputs',
        image_output_folder='./test/outputs/batch_2x',
        scaleFactor=2,
        shouldDenoiseInputImage=True,
        sharpness=None,
        devicearg='cuda',
        gpuIndex=0)
    resizeSuperXBatch.run()

    resizeSuperXBatch = ResizeSuperXBatch(
        modelFolderPath='models',
        image_input_folder='./test/inputs',
        image_output_folder='./test/outputs/batch_8x',
        scaleFactor=8,
        shouldDenoiseInputImage=True,
        sharpness=None,
        devicearg='cuda',
        gpuIndex=0)
    resizeSuperXBatch.run()




