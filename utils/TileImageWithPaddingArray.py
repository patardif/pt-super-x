from PIL import Image
from utils.TileImageWithPadding import TileImageWithPadding

class TileImageWithPaddingArray():
    '''
    Array of tile images turned into TileImageWithPadding objects.
    '''
    def __init__(self, original_tile_width: int, original_tile_height: int):
        self.originalTileWidth = original_tile_width
        self.originalTileHeight = original_tile_height
        self.tileImageWithPaddingArray = []
        pass

    def append(self,tileImageWithPadding: TileImageWithPadding):
        self.tileImageWithPaddingArray.append(tileImageWithPadding)

    def getSuperImage(self, scale: int) -> Image:
        '''
        Return the tile image enhanced/enlarged without padding.
        '''
        new_image = Image.new('RGB', (self.originalTileWidth  * scale, self.originalTileHeight  * scale))

        for tile_TileImageWithPadding in self.tileImageWithPaddingArray:
            tile_image = tile_TileImageWithPadding.getSuperResolutionImageWithoutPadding()
            tile_width, tile_height = tile_image.size

            isFirstRow = tile_TileImageWithPadding.rowIndex == 0
            isFirstCol = tile_TileImageWithPadding.colIndex == 0
            isLastRow = tile_TileImageWithPadding.isLastRow
            isLastCol = tile_TileImageWithPadding.isLastColumn

            left_inclusive = tile_TileImageWithPadding.colIndex * tile_width
            upper_inclusive = tile_TileImageWithPadding.rowIndex * tile_height

            if isFirstCol:
                left_inclusive = 0
            if isFirstRow:
                upper_inclusive = 0
            if isLastCol and not isFirstCol:
                right_exclusive = self.originalTileWidth  * scale
                left_inclusive = right_exclusive - tile_width
            if isLastRow and not isFirstRow:
                lower_exclusive = self.originalTileHeight * scale
                upper_inclusive = lower_exclusive - tile_height

            new_image.paste(tile_image, (left_inclusive, upper_inclusive))

        return new_image