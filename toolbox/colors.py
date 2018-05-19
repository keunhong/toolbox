import logging

import numpy as np
from skimage.color import rgb2lab
import skimage.io

from toolbox import caching

logger = logging.getLogger(__name__)


def visualize_color(colors, size=50):
    n_colors = colors.shape[0]
    vis = np.zeros((size, n_colors*size, 3))
    for i in range(n_colors):
        vis[:, i*size:i*size+size] = colors[i]
    return vis


def normalize_lab(lab_values):
    return (lab_values - (50, 0, 0)) / (50, 128, 128)


def denormalize_lab(norm_lab_values):
    return np.array(norm_lab_values) * (50, 128, 128) + (50, 0, 0)


def lab_rgb_gamut_bin_mask(num_bins=(10, 10, 10)):
    """
    Computes a mask of the NxMxK CIE LAB colorspace histogram that can be
    represented by the standard RGB (sRGB) color gamut.

    :param num_bins:
    :return: 3-dimensional mask.
    """
    bin_edges_L = np.linspace(0, 100, num_bins[0] + 1, endpoint=True)
    bin_edges_a = np.linspace(-90, 100, num_bins[1] + 1, endpoint=True)
    bin_edges_b = np.linspace(-110, 100, num_bins[2] + 1, endpoint=True)
    edges = (bin_edges_L, bin_edges_a, bin_edges_b)

    cache_name = \
        f'lab_rgb_gamut_bin_mask_{num_bins[0]}_{num_bins[1]}_{num_bins[2]}.png'
    cache_path = caching.get_path(cache_name)

    if caching.exists(cache_name):
        valid_bin_mask = \
            skimage.io.imread(cache_path).reshape(num_bins).astype(bool)
    else:
        print(f'Computing {cache_name}')

        rgb_gamut = np.mgrid[:255, :255, :255].reshape(3, 1, -1).transpose(
            (2, 1, 0)).astype(np.uint8)
        lab_rgb_gamut = rgb2lab(rgb_gamut)

        lab_rgb_gamut_hist, lab_rgb_gamut_hist_edges = np.histogramdd(
            lab_rgb_gamut.reshape(-1, 3),
            (bin_edges_L, bin_edges_a, bin_edges_b))

        valid_bin_mask = lab_rgb_gamut_hist > 0

        print(f'Saving {cache_name}')
        skimage.io.imsave(cache_path, valid_bin_mask.reshape(-1, 1))

    return valid_bin_mask, edges
