from PIL import Image

class TileImageWithPadding():
    '''
    Responsible to keep each tile image as PIL image with padding and provides methods to
    return tile image enhanced/enlarged with or without padding.
    '''

    def __init__(self, pil_image_with_padding: Image, tile_width: int, tile_height: int, row_index: int, col_index: int, is_last_row: bool, is_last_column: bool, padding: int):
        self.paddedimage = pil_image_with_padding
        self.tileWidth = tile_width
        self.tileHeight = tile_height
        self.rowIndex = row_index
        self.colIndex = col_index
        self.isLastRow = is_last_row
        self.isLastColumn = is_last_column
        self.padding = padding

    def getImageWithPadding(self) -> Image:
        '''
        The input low resolution image with padding
        '''
        return self.paddedimage

    def setSuperResolutionImageWithPadding(self, super_image_with_padding: Image, scale: int):
        self.superImageWithPadding = super_image_with_padding
        self.scale = scale

    def getSuperResolutionImageWithPadding(self) -> Image:
        return self.superImageWithPadding

    def getSuperResolutionImageWithoutPadding(self):
        isFirstRow = self.rowIndex == 0
        isFirstCol = self.colIndex == 0

        width, height = self.superImageWithPadding.size  # Get dimensions

        left_inclusive = self.padding * self.scale
        right_exclusive = left_inclusive + self.tileWidth * self.scale
        upper_inclusive = self.padding * self.scale
        lower_exclusive = upper_inclusive + self.tileHeight * self.scale

        if isFirstCol:
            left_inclusive = 0
            right_exclusive = self.tileWidth * self.scale

        if isFirstRow:
            upper_inclusive = 0
            lower_exclusive = self.tileHeight * self.scale

        if self.isLastRow and not isFirstRow:
            lower_exclusive = height
            upper_inclusive = lower_exclusive - self.tileHeight * self.scale

        if self.isLastColumn and not isFirstCol:
            right_exclusive = width
            left_inclusive = right_exclusive - self.tileWidth * self.scale

        tile = self.superImageWithPadding.crop((left_inclusive, upper_inclusive, right_exclusive, lower_exclusive))
        return tile