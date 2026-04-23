import os

class ImageSplitter:
    def __init__(self, image_path, output_dir, split_size=512):
        self.image_path = image_path
        self.output_dir = output_dir
        self.split_size = split_size

    def split_image(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        image = Image.open(self.image_path)
        width, height = image.size

        for i in range(0, width, self.split_size):
            for j in range(0, height, self.split_size):
                box = (i, j, min(i + self.split_size, width), min(j + self.split_size, height))
                split_image = image.crop(box)
                split_image.save(os.path.join(self.output_dir, f'split_{i}_{j}.png'))