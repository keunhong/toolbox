import logging
import math
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
from scipy import misc
from scipy.ndimage.interpolation import zoom
from scipy.stats import wasserstein_distance
from skimage import transform

from rendkit import pfm
from toolbox.stats import find_outliers

logger = logging.getLogger(__name__)

BoundingBox = Tuple[int, int, int, int]

QUAL_COLORS = [
    (166, 206, 227),
    (31, 120, 180),
    (178, 223, 138),
    (51, 160, 44),
    (251, 154, 153),
    (227, 26, 28),
    (253, 191, 111),
    (255, 127, 0),
    (202, 178, 214),
    (106, 61, 154),
    (255, 255, 153),
    (177, 89, 40),
    (141, 211, 199),
    (255, 255, 179),
    (190, 186, 218),
    (251, 128, 114),
    (128, 177, 211),
    (253, 180, 98),
    (179, 222, 105),
    (252, 205, 229),
    (217, 217, 217),
    (188, 128, 189),
    (204, 235, 197),
    (255, 237, 111),
]


def is_img(path):
    img_types = ['png', 'tiff', 'tif', 'jpg', 'gif', 'jpeg']
    for t in img_types:
        if str(path).lower().endswith(t):
            return True
    return False


def compute_segment_median_colors(image: np.ndarray, segment_mask: np.ndarray):
    num_segments = sum((1 for i in np.unique(segment_mask) if i >= 0))

    segment_colors = []
    for segment_id in range(num_segments):
        segment_pixels = image[segment_mask == segment_id]
        if len(segment_pixels) > 0:
            median_color = np.median(segment_pixels, axis=0)
            segment_colors.append((median_color[0],
                                   median_color[1],
                                   median_color[2]))
            logger.info('segment {}: {} pixels, median_color={}'.format(
                segment_id, len(segment_pixels), repr(median_color)))
        else:
            segment_colors.append((1, 0, 1))
            logger.info('segment {}: not visible.'.format(segment_id))

    return np.array(segment_colors)


def compute_mask_bbox(mask: np.ndarray) -> BoundingBox:
    """
    Computes bounding box which contains the mask.
    :param mask:
    :return:
    """
    yinds, xinds = np.where(mask)
    xmin, xmax = np.min(xinds), np.max(xinds)
    ymin, ymax = np.min(yinds), np.max(yinds)

    return ymin, ymax, xmin, xmax


def crop(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    ymin, ymax, xmin, xmax = bbox
    return image[ymin:ymax, xmin:xmax]


def pad(image: np.ndarray,
        pad_width: int,
        mode='constant',
        fill=0) -> np.ndarray:
    if len(image.shape) == 4:
        padding = ((pad_width,), (pad_width,), (0,), (0,))
    elif len(image.shape) == 3:
        padding = ((pad_width,), (pad_width,), (0,))
    elif len(image.shape) == 2:
        padding = ((pad_width,), (pad_width,))
    else:
        raise RuntimeError("Unsupported image shape {}".format(image.shape))

    return np.pad(image, pad_width=padding, mode=mode, constant_values=fill)


def rotate(image: np.ndarray, angle: float, crop=False) -> np.ndarray:
    rotated = transform.rotate(image, angle)
    if crop:
        radius = min(image.shape[:2]) / 2.0
        length = math.sqrt(2) * radius
        height, width = image.shape[:2]
        rotated = rotated[
                  height/2-length/2:height/2+length/2,
                  width/2-length/2:width/2+length/2]
    return rotated


def apply_mask(image, mask, fill=0):
    """
    Fills pixels outside the mask with a constant value.
    :param image: to apply the mask to.
    :param mask: binary mask with True values for pixels that are to be preserved.
    :param fill: fill value.
    :return: Masked image
    """
    masked = image.copy()
    masked[~mask] = fill
    return masked


def normalize(image, low=0.0, high=1.0):
    """
    Normalized the image to a range.
    :param image:
    :param low: lowest value after normalization.
    :param high: highest value after normalization.
    :return:
    """
    image_01 = (image - image.min()) / (image.max() - image.min())
    return image_01 * (high - low) + low


def rgb2gray(image):
    return (0.2125 * image[:, :, 0]
            + 0.7154 * image[:, :, 1]
            + 0.0721 * image[:, :, 2])


def trim_image(image, mask):
    y, x = np.where(mask)
    return image[y.min():y.max(), x.min():x.max()]


def suppress_outliers(image, thres=3.5, preserve_dark=True):
    new_map = image.copy()
    outliers = find_outliers(np.reshape(image, (-1, 3)), thres=thres)\
        .reshape(image.shape[:2])
    med = np.median(image, axis=(0, 1))
    if preserve_dark:
        outliers &= np.any(image > med, axis=2)
    new_map[outliers] = med
    return new_map


def resize(array, shape, order=3):
    if len(array.shape) != 2 and len(array.shape) != 3:
        raise RuntimeError("Input array must have 2 or 3 dimensions but {} "
                           "were given.".format(len(array.shape)))
    if isinstance(shape, float):
        scales = (shape, shape, 1)
    elif isinstance(shape, tuple):
        scales = (shape[0] / array.shape[0],
                  shape[1] / array.shape[1], 1)
    else:
        raise RuntimeError("shape must be tuple or float.")

    n_channels = 1 if len(array.shape) == 2 else array.shape[2]
    if n_channels == 1:
        scales = scales[:2]
    output = zoom(array, scales, order=order)
    output = output.astype(dtype=array.dtype)
    return output.clip(array.min(), array.max())


def save_image(path, array):
    if array.dtype == np.uint8:
        array = array.astype(dtype=float)
        array /= 255.0
    array = np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(dtype=np.uint8)
    misc.imsave(path, array)


def load_image(path, mode='RGB'):
    image = misc.imread(path, mode=mode)
    image = image.astype(dtype=np.float32) / 255.0
    return image


def reinhard(image_hdr, thres):
    return image_hdr * (1 + image_hdr / thres ** 2) / (1 + image_hdr)


def reinhard_inverse(image_ldr, thres):
    Lw = thres
    Ld = image_ldr
    rt = np.sqrt(Lw) * np.sqrt(Lw * (Ld - 1)**2 + 4 * Ld)
    return Lw * (Ld - 1) + rt


def bright_pixel_mask(image, percentile=80):
    perc = np.percentile(np.unique(image[:, :, :3].min(axis=2)), percentile)
    mask = np.all(image < perc, axis=-1)
    return mask


def compute_fg_bbox(image):
    bbox = mask_bbox(bright_pixel_mask(image))
    return bbox


def bbox_shape(bbox):
    return bbox[1] - bbox[0], bbox[3] - bbox[2]


def bbox_centroid(bbox):
    height, width = bbox_shape(bbox)
    return bbox[0] + height / 2, bbox[2] + width / 2


def bbox_make_square(bbox):
    height, width = bbox_shape(bbox)
    maxlen = max(height, width)
    cy, cx = bbox_centroid(bbox)
    bbox = (cy - maxlen / 2, cy + maxlen / 2,
            cx - maxlen / 2, cx + maxlen / 2)
    bbox = tuple(int(i) for i in bbox)
    return bbox


def crop_tight_fg(image, shape=None, bbox=None, fill=1.0, order=3,
                  use_pil=False):
    if bbox is None:
        bbox = compute_fg_bbox(image)
    image = crop_bbox(image, bbox)

    height, width = image.shape[:2]
    max_len = max(height, width)
    if shape is None:
        shape = (max_len, max_len)

    output_shape = (max_len, max_len)
    if len(image.shape) > 2:
        output_shape += (image.shape[-1],)
    output = np.full(output_shape, fill, dtype=image.dtype)
    if height > width:
        lo = (height - width) // 2
        output[:, lo:lo + width] = image
    else:
        lo = (width - height) // 2
        output[lo:lo + height, :] = image
    if use_pil:
        output = transform.resize(output, shape, anti_aliasing=True,
                                  preserve_range=True,
                                  mode='constant', cval=fill)
    else:
        output = resize(output, shape, order=order)
    output = np.clip(output, image.min(), image.max())
    return output.squeeze()


def save_arr(path, arr):
    np.savez(path, arr)


def load_arr(path):
    return np.load(path)['arr_0'][()]


def save_hdr(path: Path, image):
    if isinstance(path, str):
        path = Path(path)
    ext = path.suffix[1:]
    if ext == 'exr':
        if len(image.shape) == 3:
            image = image[:, :, [2, 1, 0]]
        cv2.imwrite(str(path), image)
    elif ext == 'pfm':
        pfm.pfm_save(str(path))
    else:
        raise RuntimeError("Unknown format {}".format(ext))


def load_hdr(path: Union[Path, str], ext=None):
    if isinstance(path, str):
        path = Path(path)

    if ext is None:
        ext = path.suffix[1:]
    if ext in {'exr', 'hdr'}:
        import cv2
        im = cv2.imread(str(path), -1)
        if len(im.shape) == 3:
            im = im[:, :, [2, 1, 0]]
    elif ext == 'pfm':
        im = pfm.pfm_read(path)
    else:
        raise RuntimeError("Unknown format {}".format(ext))
    return im


def crop_bbox(image, bbox):
    return image[bbox[0]:bbox[1], bbox[2]:bbox[3]]


def mask_bbox(mask):
    yinds, xinds = np.where(mask)
    bbox = np.min(yinds), np.max(yinds), np.min(xinds), np.max(xinds)
    return tuple(int(i) for i in bbox)


def visualize_map(image, bg_value=-1, return_legends=False):
    assert len(image.shape) == 2
    output = np.ones((*image.shape, 3))
    values = [v for v in np.unique(image) if v != bg_value]
    if len(values) > len(QUAL_COLORS):
        logger.warning('Qualitative colors will wrap around since there are '
                       '%d values to map.', len(values))

    legends = {}

    for i, value in enumerate(values):
        color = QUAL_COLORS[i % len(QUAL_COLORS)]
        output[image == value] = (np.array(color) / 255.0
                                  if value != bg_value
                                  else (0, 0, 0))
        if return_legends and value != bg_value:
            legends[value] = color

    if return_legends:
        return output, legends
    return output


def compute_color_histogram(pixels, n_bins=256, normalized=True):
    r_hist, _ = np.histogram(pixels[:, 0], bins=n_bins, density=normalized)
    g_hist, _ = np.histogram(pixels[:, 1], bins=n_bins, density=normalized)
    b_hist, _ = np.histogram(pixels[:, 2], bins=n_bins, density=normalized)
    return np.stack((r_hist, g_hist, b_hist))


def color_wasserstein_dist(hist1, hist2):
    weights = [0.2, 0.4, 0.4]
    return sum(w * wasserstein_distance(hist1[i], hist2[i]) for i, w in
               zip([0, 1, 2], weights))


def linear_to_srgb(linear):
    srgb = linear.copy()
    less = linear <= 0.0031308
    srgb[less] = linear[less] * 12.92
    srgb[~less] = 1.055 * np.power(linear[~less], 1.0 / 2.4) - 0.055
    return srgb


def srgb_to_linear(srgb):
    if srgb.dtype == np.uint8:
        srgb = np.float32(srgb) / 255.0
    linear = srgb
    less = linear <= 0.4045
    linear[less] = linear[less] / 12.92
    linear[~less] = np.power((linear[~less] + 0.055) / 1.055, 2.4)
    return linear


def to_8bit(image):
    return np.clip(image * 255, 0, 255).astype('uint8')

