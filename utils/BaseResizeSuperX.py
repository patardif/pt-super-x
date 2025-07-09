from abc import ABC, abstractmethod
import torch
from torch import nn
from PIL import Image

# abstract base class for ResizeSuperX
class BaseResizeSuperX(ABC):

    @abstractmethod
    def apply_model(self, model: nn.Module, img: Image):
        pass  # Abstract method, must be implemented in subclasses