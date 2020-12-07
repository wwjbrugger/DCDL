"""
In greyscale pictures high values are represented by white
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def visualize_pic(pic_array, label_array, class_names, title, colormap):
    """ show 10 first  pictures """
    for i in range(20):
        plt.subplot(5, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if pic_array.shape[3] == 1:
            plt.imshow(pic_array[i,:,:,0], cmap=colormap)
        elif pic_array.shape[3] == 3:
            plt.imshow(pic_array[i], cmap=colormap)
        else:
            raise ValueError('Picture should have 1 or 3 channels not'.format(pic_array.shape[3]))
        plt.xlabel(class_names[np.argmax(label_array[i])])

    st = plt.suptitle(title, fontsize=14)
    st.set_y(1)
    plt.tight_layout()
    plt.show()


def dither_pic(pic_array, values_max_1):
    """ dither pictures """
    for channel in range(pic_array.shape[3]):
        for i, pic in enumerate(pic_array[:, :, :, channel]):
            if values_max_1:
                # if pictures are scaled into the range of 0 to 1
                picture_grey = Image.fromarray(pic * 255)
            else:
                # if pictures are in range  0 to 255
                picture_grey = Image.fromarray(pic)
            # dither picture
            picture_dither = picture_grey.convert("1")
            picture_dither_np = np.array(picture_dither)
            # set 0 values to -1
            pic_array[i,:,:,channel] = np.where(picture_dither_np, 1, -1)
    return pic_array


