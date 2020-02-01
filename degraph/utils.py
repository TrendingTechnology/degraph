from .config import TF_FLOAT, EPSILON
import tensorflow as tf
import numpy as np
import matplotlib.cm
import matplotlib.colors as mcolors
from typing import Tuple, Union, Any
import sys


def get_cmap(name: str = 'jet', size: int = 256):
    """
    Return a numpy array containing the selected colormap. The expected shape of the array is (size, 3) and
    the entries are floating point values in the range [0, 1]. This is meant to be used in conjunction with
    Tensorflow's gather operator.
    :param name: The name of the colormap (from the matplotlib catalog)
    :param size: The number of entries required
    :return: A numpy array
    """
    assert isinstance(size, int)
    cm = matplotlib.cm.get_cmap(name, lut=size)
    output = cm(np.arange(size))[:, :3]
    return output


def create_palette(lightness_range: Tuple[float, float] = (0., 0.9)) -> np.ndarray:
    """
    Create a palette of colors whose lightness is in the given interval. The returned array is compatible with Cairo.
    :param lightness_range: The lightness range for the colors.
    :return: A numpy array of expected shape [color_count, 3].
    """
    lightness_range = np.amin(lightness_range), np.amax(lightness_range)
    output = [mcolors.to_rgb(color[1]) for color in mcolors.CSS4_COLORS.items()]
    output = map(lambda value: (np.asarray(value), mcolors.rgb_to_hsv(value)), output)
    output = map(np.concatenate, output)
    output = np.stack(list(output))
    output = np.array(output, dtype=np.float32)
    output = output[(output[:, -1] >= lightness_range[0]) & (output[:, -1] <= lightness_range[1])]
    return output[:, :3]


def apply_colormap(img: tf.Tensor, range_mode: str = 'normalize',
                   colormap: Union[str, None, tf.Tensor, np.ndarray] = None) -> tf.Tensor:
    if range_mode not in ('normalize', 'saturate'):
        raise ValueError(f'Unrecognized range_mode: {range_mode}')

    assert tf.rank(img) == 2 and img.dtype == TF_FLOAT

    f = 255.0
    if range_mode == 'normalize':
        range_ = tf.reduce_min(img), tf.reduce_max(img)
        img -= range_[0]
        f = 255.0 * tf.math.reciprocal((range_[1] - range_[0]) + EPSILON)

    img = tf.saturate_cast(img * f, dtype=tf.uint8)
    if colormap is None:
        img = tf.tile(tf.expand_dims(img, -1), [1, 1, 3])
    else:
        if isinstance(colormap, str):
            colormap = get_cmap(colormap)
        colormap = tf.convert_to_tensor(colormap)
        img = tf.gather(colormap, tf.cast(img, tf.int32))
        img = tf.saturate_cast(img * 255.0, tf.uint8)
    return img


def join_ident(prefix: Union[str, Any], name: str) -> str:
    if not isinstance(prefix, str):
        if hasattr(prefix, 'name'):
            prefix = prefix.name
        else:
            raise ValueError(f'Unsupported prefix object type: {type(prefix)}')
    return f'{prefix}.{name}'


def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn

