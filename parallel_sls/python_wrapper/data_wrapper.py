import numpy as np


def binary_to_packed_uint8(binary_array):
    not_flattend_packed = np.packbits(binary_array, axis=1)
    flattend_packed = not_flattend_packed.flatten()
    return flattend_packed


def binary_to_packed_uint8_continguous(binary_array):
    return np.ascontiguousarray(binary_to_packed_uint8(binary_array), dtype = np.uint8)


def uint8_to_unpacked_binary(uint8_array, vars_per_vector):
    flattend_unpacked = np.unpackbits(uint8_array)
    unflattend_unpacked = np.reshape(flattend_unpacked, (-1, vars_per_vector))
    return unflattend_unpacked
