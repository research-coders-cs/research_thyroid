import math
import matplotlib.pyplot as plt
from typing import List, Tuple, NoReturn

Sample = List[Tuple[object, str]]
Size = Tuple[int, int]


def display_sample(sample: Sample, img_size: Size=(3, 3)) -> NoReturn:
    """
    Display sample image for inspection.
    :param sample: list of Sample
    :param img_size: the size of image in inches unit
    """
    image_per_row = 8
    number_of_row = math.ceil(len(sample) / image_per_row)

    pane_size = (img_size[0] * image_per_row, img_size[1] * number_of_row)

    fig, m_axs = plt.subplots(number_of_row, image_per_row, figsize=pane_size)

    for ((c_x, c_y), c_ax) in zip(sample, m_axs.flatten()):
        # Make it grayscale
        img = c_x.permute(1, 2, 0)
        img = img.mean(axis=2)
        c_ax.imshow(img, cmap='gray')

        c_ax.set_title(f"{c_y}")
        c_ax.axis('off')
