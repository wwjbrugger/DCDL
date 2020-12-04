#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`diffusion`
=======================

.. moduleauthor:: hbldh <henrik.blidh@swedwise.com>
Created on 2016-09-12, 11:34

"""

import numpy as np
from tqdm import tqdm


_DIFFUSION_MAPS = {
    'floyd-steinberg': (
        (1, 0,  7 / 16),
        (-1, 1, 3 / 16),
        (0, 1,  5 / 16),
        (1, 1,  1 / 16)
    ),
    'atkinson': (
        (1, 0,  1 / 8),
        (2, 0,  1 / 8),
        (-1, 1, 1 / 8),
        (0, 1,  1 / 8),
        (1, 1,  1 / 8),
        (0, 2,  1 / 8),
    ),
    'jarvis-judice-ninke': (
        (1, 0,  7 / 48),
        (2, 0,  5 / 48),
        (-2, 1, 3 / 48),
        (-1, 1, 5 / 48),
        (0, 1,  7 / 48),
        (1, 1,  5 / 48),
        (2, 1,  3 / 48),
        (-2, 2, 1 / 48),
        (-1, 2, 3 / 48),
        (0, 2,  5 / 48),
        (1, 2,  3 / 48),
        (2, 2,  1 / 48),
    ),
    'stucki': (
        (1, 0,  8 / 42),
        (2, 0,  4 / 42),
        (-2, 1, 2 / 42),
        (-1, 1, 4 / 42),
        (0, 1,  8 / 42),
        (1, 1,  4 / 42),
        (2, 1,  2 / 42),
        (-2, 2, 1 / 42),
        (-1, 2, 2 / 42),
        (0, 2,  4 / 42),
        (1, 2,  2 / 42),
        (2, 2,  1 / 42),
    ),
    'burkes': (
        (1, 0,  8 / 32),
        (2, 0,  4 / 32),
        (-2, 1, 2 / 32),
        (-1, 1, 4 / 32),
        (0, 1,  8 / 32),
        (1, 1,  4 / 32),
        (2, 1,  2 / 32),
    ),
    'sierra3': (
        (1, 0,  5 / 32),
        (2, 0,  3 / 32),
        (-2, 1, 2 / 32),
        (-1, 1, 4 / 32),
        (0, 1,  5 / 32),
        (1, 1,  4 / 32),
        (2, 1,  2 / 32),
        (-1, 2, 2 / 32),
        (0, 2,  3 / 32),
        (1, 2,  2 / 32),
    ),
    'sierra2': (
        (1, 0,  4 / 16),
        (2, 0,  3 / 16),
        (-2, 1, 1 / 16),
        (-1, 1, 2 / 16),
        (0, 1,  3 / 16),
        (1, 1,  2 / 16),
        (2, 1,  1 / 16),
    ),
    'sierra-2-4a': (
        (1, 0,  2 / 4),
        (-1, 1, 1 / 4),
        (0, 1,  1 / 4),
    ),
    'stevenson-arce': (
        (2, 0,   32 / 200),
        (-3, 1,  12 / 200),
        (-1, 1,  26 / 200),
        (1, 1,   30 / 200),
        (3, 1,   30 / 200),
        (-2, 2,  12 / 200),
        (0, 2,   26 / 200),
        (2, 2,   12 / 200),
        (-3, 3,   5 / 200),
        (-1, 3,  12 / 200),
        (1, 3,   12 / 200),
        (3, 3,    5 / 200)
    )
}
def error_diffusion_dithering(image, method='floyd-steinberg'):
    dither_image = []
    for pic in tqdm(image):
        dither_image.append(error_diffusion_dithering_single_picture(np.array(pic),method=method))
    #dither_image =  np.array([error_diffusion_dithering_single_picture(np.array(pic), method) for pic in image])
    return np.array(dither_image)

def error_diffusion_dithering_single_picture(image, method='floyd-steinberg'):
    dither_image = np.zeros(image.shape)
    for chanel in range(image.shape[-1]):
        dither_chanel = error_diffusion_dithering_single_channel(np.array(image[:,:,chanel]), method)
        dither_image[:,:,chanel] = dither_chanel
    return dither_image

def error_diffusion_dithering_single_channel(image_origin, method='floyd-steinberg'):
    image = image_origin.copy()
    diff_map = _DIFFUSION_MAPS.get(method.lower())

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            old_pixel = image[y, x] # only this line 3,51
            old_pixel = 0 if old_pixel< 0.0 else old_pixel # [old_pixel < 0.0] = 0.0 # + 3 sec
            old_pixel = 1 if old_pixel > 1 else old_pixel# [old_pixel > 255.0] = 255.0 # + 3sec
            new_pixel = np.around(old_pixel) # 5,3
            quantization_error = old_pixel - new_pixel
            image[y, x] = new_pixel #17 s
            for dx, dy, diffusion_coefficient in diff_map: # 88,42  90,31
                xn, yn = x + dx, y + dy
                if (0 <= xn < image.shape[1]) and (0 <= yn < image.shape[0]):
                    image[yn, xn] += quantization_error * diffusion_coefficient
    return image
