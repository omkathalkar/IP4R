from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from PIL import Image
import numpy as np
import torch

class MnistCNNImageProcessor(BaseImageProcessor):
    model_input_names = ['pixel_values']

    def __init__(self, do_resize=True, size=None, do_rescale=True, rescale_factor=1./255, do_normalize=False, do_convert_grayscale=True, **kwargs):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size or {'height': 28, 'width': 28}
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.do_convert_grayscale = do_convert_grayscale

    def preprocess(self, images, return_tensors='pt', **kwargs):
        if not isinstance(images, list):
            images = [images]
        processed = []
        for img in images:
            if not isinstance(img, Image.Image):
                # try numpy -> PIL
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                else:
                    # try torch tensor -> numpy -> PIL
                    if torch.is_tensor(img):
                        img = img.detach().cpu().numpy()
                        if img.ndim == 3 and img.shape[0] in (1, 3):
                            img = np.transpose(img, (1, 2, 0))
                        img = Image.fromarray(img.astype('uint8'))
                    else:
                        raise ValueError('Unsupported image type for preprocess')
            if self.do_convert_grayscale:
                img = img.convert('L')
            if self.do_resize:
                img = img.resize((self.size['width'], self.size['height']))
            arr = np.array(img).astype('float32')
            if self.do_rescale:
                arr = arr * self.rescale_factor
            # (H, W) -> (1, H, W) channel-first grayscale
            arr = np.expand_dims(arr, 0)
            processed.append(arr)
        tensor = torch.tensor(np.stack(processed, axis=0))  # (B, 1, H, W)
        return BatchFeature(data={'pixel_values': tensor}, tensor_type='pt')

    def to_dict(self):
        output = super().to_dict()
        output.update({
            'do_resize': self.do_resize,
            'size': self.size,
            'do_rescale': self.do_rescale,
            'rescale_factor': self.rescale_factor,
            'do_normalize': self.do_normalize,
            'do_convert_grayscale': self.do_convert_grayscale,
            'image_processor_type': self.__class__.__name__,
        })
        return output
