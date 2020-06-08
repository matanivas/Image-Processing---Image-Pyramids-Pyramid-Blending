import math
import imageio
from matplotlib import pyplot as plt
import scipy.signal
import scipy.io.wavfile as wav
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
import numpy as np
from imageio import imread, imwrite
from skimage.color import rgb2gray
import scipy.misc
import os

GREY = 1
NORMALIZATION_FACTOR = 2
MIN_IM_DIM = 16

"""
this function gets as parameters image path and its representation and read the image at the specified representation"""
def read_image(filename, representation):
    im_float = imageio.imread(filename)
    type = im_float.dtype
    if type == int or type == np.uint8:
        im_float = im_float.astype(np.float64) / 255
    if representation == GREY:
        return rgb2gray(im_float)
    return im_float


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)

def gaus_blurr(im, gaus_filter):
    blurred = scipy.signal.convolve2d(im, gaus_filter, "same")
    x =  scipy.signal.convolve2d(blurred, gaus_filter.T, "same")
    return x


def expand_im(im, gaus_filter):
    x, y = im.shape
    padded_im = np.zeros((x * 2, y * 2))
    padded_im[::2, ::2] = im
    return gaus_blurr(padded_im, gaus_filter) * 2


def reduce_im(im, gaus_filter):
    blurred = gaus_blurr(im, gaus_filter)
    return blurred[1::2, 1::2]

def generate_filter(filter_size):
    conv = np.array([[1, 1]])
    in_process = np.array([[1, 1]]) / NORMALIZATION_FACTOR
    for i in range(filter_size - 2):
        in_process = scipy.signal.convolve2d(in_process, conv) / NORMALIZATION_FACTOR
    return in_process


def build_gaussian_pyramid(im, max_levels, filter_size):
    min_dim = np.min(im.shape)
    gaus_filter = generate_filter(filter_size)
    # print("max level is: " , max_levels, "shape is: " , im.shape)
    cur_level = im
    pyr = [im]
    for i in range(max_levels - 1):
        if (min_dim / 2) < MIN_IM_DIM:
            break
        min_dim /= 2
        cur_level = reduce_im(cur_level, gaus_filter)
        pyr.append(cur_level)
    # print("len is: ", len(pyr))
    return pyr, gaus_filter

def build_laplacian_pyramid(im, max_levels, filter_size):
    pyr, gaus_filter = build_gaussian_pyramid(im, max_levels, filter_size)
    laplac_pyr = []
    for i in range(len(pyr) - 1):
        exp = expand_im(pyr[i + 1], gaus_filter)
        laplac_pyr.append(pyr[i] - expand_im(pyr[i + 1], gaus_filter))
    laplac_pyr.append(pyr[len(pyr) - 1])
    return laplac_pyr, gaus_filter

def laplacian_to_image(lpyr, filter_vec, coeff):
    weighted_lpyr = coeff * np.array(lpyr)

    for i in reversed(range(1, len(weighted_lpyr), 1)):
        weighted_lpyr[i-1] += expand_im(weighted_lpyr[i], filter_vec)
    return weighted_lpyr[0]

def stretch_im(im):
    min_value = np.amin(im)
    max_value = np.amax(im)
    return (im - min_value) / (max_value - min_value)


def render_pyramid(pyr, levels):
    normalized_pyr = []
    out_width = 0
    levels = min(levels, len(pyr))
    for i in range(levels):
        out_width += pyr[i].shape[1]
        normalized_pyr.append(stretch_im(pyr[i]))
    res = np.zeros((pyr[0].shape[0], out_width))
    position = 0
    for i in range(levels):
        res[0:pyr[i].shape[0], position: position + pyr[i].shape[1]] += normalized_pyr[i]
        position += pyr[i].shape[1]
    return res

def display_pyramid(pyr, levels):
    im_pyr = render_pyramid(pyr, levels)
    plt.imshow(im_pyr, cmap='gray')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
        mask = mask.astype(float)
        L1, filter = build_laplacian_pyramid(im1, max_levels, filter_size_im)
        L2, filter = build_laplacian_pyramid(im2, max_levels, filter_size_im)
        Gm, filter = build_gaussian_pyramid(mask, max_levels, filter_size_mask)
        levels = min(len(L1), len(L2), len(Gm))
        LBlended = []
        for i in range(levels):
            LBlended.append(Gm[i] * L1[i] + (1 - Gm[i]) * L2[i])
        return laplacian_to_image(LBlended, filter, np.ones(levels))


def colored_blending(path1, path2, pathM, levels, filter_size_im, filter_size_mask):
    mask = read_image(pathM, 1)
    mask = mask.astype(bool)
    im1 = read_image(path1, 2)
    im2 = read_image(path2, 2)
    im_blendedR = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, levels, filter_size_im, filter_size_mask)
    im_blendedG = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, levels, filter_size_im, filter_size_mask)
    im_blendedB = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, levels, filter_size_im, filter_size_mask)
    im_blendedRGB = np.stack((im_blendedR, im_blendedG, im_blendedB), axis=2)
    np.clip(im_blendedRGB, 0, 1)
    return im1, im2, mask, im_blendedRGB

def blending_example1():
    im1, im2, mask, im_blendedRGB = colored_blending(relpath('externals/TEA.jpg'),
                                                     relpath('externals/STORM.jpg'),
                                                     relpath('externals/maskTEA.jpg'), 5, 5, 3)
    return im1, im2, mask, im_blendedRGB

def blending_example2():
    im1, im2, mask, im_blendedRGB = colored_blending(relpath('externals/red_sea.jpg'),
                                                     relpath('externals/bibi.jpg'),
                                                     relpath('externals/bibiMask.jpg'), 3, 3, 3)
    return im1, im2, mask, im_blendedRGB


def plots(image1, image2, mask, blended):
    grbg, ax = plt.subplots(2,2)
    ax[0,0].imshow(image1)
    ax[0,1].imshow(image2)
    ax[1,0].imshow(mask, cmap='gray')
    ax[1,1].imshow(blended)
    plt.show()

im1_1, im2_1, mask_1, im_blend_1 = blending_example1()
im1_2, im2_2, mask_2, im_blend_2 = blending_example2()

plots(im1_1, im2_1, mask_1, im_blend_1)
plots(im1_2, im2_2, mask_2, im_blend_2)
