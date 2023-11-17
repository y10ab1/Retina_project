import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image

class CenterCropTransform:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.transform = transforms.CenterCrop((height, width))

    def __call__(self, image):
        return self.transform(image)

class CustomCropTransform:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def __call__(self, image):
        return image[:, self.y_min:self.y_max, self.x_min:self.x_max]

class RetinalCrop:
    def __init__(self):
        self.cropper_alpha = CenterCropTransform(width=1424, height=1424)
        self.cropper_beta = CenterCropTransform(width=1536, height=1536)
        self.cropper_gamma = CustomCropTransform(x_min=248, x_max=3712, y_min=408, y_max=3872)

    def pad_to_square(self, image):
        _, height, width = image.shape
        max_side = max(height, width)
        padding_left = (max_side - width) // 2
        padding_top = (max_side - height) // 2
        padding_right = max_side - width - padding_left
        padding_bottom = max_side - height - padding_top
        return F.pad(image, [padding_left, padding_top, padding_right, padding_bottom])

    def transform(self, image):
        image = self.pad_to_square(image)
        if image.shape == (3, 2144, 2144):
            image_cropped = self.cropper_alpha(image)
        elif image.shape == (3, 2048, 2048):
            image_cropped = self.cropper_beta(image)
        elif image.shape == (3, 4288, 4288):
            image_cropped = self.cropper_gamma(image)
        else:
            raise ValueError("Unsupported image shape")
        return image_cropped

if __name__ == '__main__':
    # Create a test instance of RetinalCrop
    retinal_crop = RetinalCrop()

    # Load image
    test_image = Image.open('/home/yuehpo/coding/Retina_project/dataset/Training_Set/Training_Set/Training/1757.png')
    test_image = F.to_tensor(test_image)

    # Apply transformation
    transformed_image = retinal_crop.transform(test_image)

    print("Original image shape:", test_image.shape)
    print("Transformed image shape:", transformed_image.shape)

    # Save transformed image
    transformed_image = F.to_pil_image(transformed_image)
    transformed_image.save('9_transformed.png')