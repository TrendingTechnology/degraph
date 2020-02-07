from .config import TF_FLOAT, EPSILON
from .types import RankStaticShape, RBF

from typing import Union, Callable, List, Sequence
import warnings

import tensorflow as tf
import numpy as np


def reduce_any_nan(t: tf.Tensor) -> tf.Tensor:
    return tf.reduce_any(tf.math.is_nan(t))


def tf_repeat_1d(x: tf.Tensor, t) -> tf.Tensor:
    tf.assert_rank(x, 1)
    tf.assert_rank(t, 0)
    return tf.reshape(tf.tile(tf.expand_dims(x, axis=-1), [1, t]), shape=[x.shape[0] * t])


def tri_ones_lower(k):
    return tf.linalg.band_part(tf.ones((k, k)), -1, 0)


def tri_ones_upper(k):
    return tf.linalg.band_part(tf.ones((k, k)), 0, -1)


def mish(x: tf.Tensor) -> tf.Tensor:
    return x * tf.math.tanh(tf.math.log(1 + tf.exp(x)))


def cosine_decay_factor(step, step_count, final_factor) -> tf.Tensor:
    factor = tf.cast(tf.minimum(step, step_count), dtype=TF_FLOAT) / step_count
    factor = (1 + tf.math.cos(np.pi * factor))/2.
    factor = (1. - final_factor) * factor + final_factor
    return factor


def tf_binom(n: tf.Tensor, k: tf.Tensor):
    """
    Binomial coefficient (n k). Note this should be used for modest values of N since the precision is limited to int64.
    :param n:
    :param k:
    :return:
    """
    # TODO assert
    # TODO create function binom_all
    k = tf.minimum(k, n-k)
    v1 = tf.cast(tf.reduce_prod(tf.range(n-k+1, n+1)), dtype=tf.int64)  # Note tf.reduce_prod([]).dtype == tf.float32
    v2 = tf.cast(tf.reduce_prod(tf.range(1, k+1)), dtype=tf.int64)
    return tf.math.floordiv(v1, v2)


def hgrid_normalize(shape: RankStaticShape, grid: tf.Tensor):   # TODO rename and normalize in the interval [-1, 1]
    factor = tf.cast(shape, dtype=TF_FLOAT) - 1
    offset = tf.where(factor > 0, x=0., y=0.5)  # Single element axis is centred
    factor = tf.where(factor > 0, x=factor, y=1.)
    factor = tf.math.reciprocal(factor)
    return tf.cast(grid, dtype=TF_FLOAT) * factor + offset


def normalize(tensor: tf.Tensor) -> tf.Tensor:
    limits = tf.reduce_min(tensor), tf.reduce_max(tensor)
    return (tensor - limits[0])/(limits[1] - limits[0] + EPSILON)


def hsphere_coords(shape: RankStaticShape) -> tf.Tensor:
    center = (tf.convert_to_tensor(shape, dtype=TF_FLOAT) - 1.)/2.0
    sq_radius = tf.square(tf.cast(tf.reduce_max(shape), dtype=TF_FLOAT)/2.)     # Use the maximum side for the radius
    coords = tf_hgrid_coords(shape=shape)
    sq_dists = tf.reduce_sum(tf.square(tf.cast(coords, dtype=TF_FLOAT) - center), axis=-1)
    return tf.squeeze(tf.gather(coords, indices=tf.where(sq_dists <= sq_radius)), axis=1)


def tf_hgrid_coords(shape: RankStaticShape) -> tf.Tensor:
    """
    Produce hyper-grid coordinates
    :param shape:
    :return:
    """
    static_rank = len(shape)
    shape = tf.convert_to_tensor(shape, dtype=tf.int64)
    comps = [tf.tile(tf_repeat_1d(tf.range(shape[k]), t=tf.reduce_prod(shape[k+1:])), [tf.reduce_prod(shape[:k])])
             for k in range(static_rank)]
    return tf.transpose(tf.stack(comps))


def crbf_base(sq_d: tf.Tensor, sq_r: tf.Tensor, core: Callable) -> tf.Tensor:
    """
    Base implementation for compactly supported radial basis functions.
    :param sq_d:
    :param sq_r: The squared radius of the RBF, expected a positive real.
    :param core:
    :return:
    """
    orig_shape = sq_d.shape
    sq_d = tf.reshape(sq_d, [-1])
    tf.assert_rank(sq_d, 1)

    idxs = tf.squeeze(tf.where(sq_d < sq_r), axis=-1)
    sq_d1 = tf.gather(sq_d, indices=idxs)
    r = tf.sqrt(sq_d1 / sq_r)     # Normalized radius
    r = core(r)
    ret = tf.tensor_scatter_nd_add(tf.zeros_like(sq_d), indices=tf.expand_dims(idxs, -1), updates=r)
    return tf.reshape(ret, shape=orig_shape)


def crbf_bump_scaled(sq_d: tf.Tensor, sq_r: tf.Tensor) -> tf.Tensor:
    """
    Bump function, the function is scaled so that the maximum at r=0 has value 1.
    :param sq_d:
    :param sq_r:
    :return:
    """
    return crbf_base(sq_d=sq_d, sq_r=sq_r, core=lambda r: tf.exp(-1./(1 - tf.square(r)))/tf.exp(-1.))


def crbf_wendland_d3c2(sq_d: tf.Tensor, sq_r: tf.Tensor) -> tf.Tensor:
    """
    Wendland compactly supported radial basis function for up to 3 dimensions and continuity to the second derivative.
    :param sq_d:
    :param sq_r: Squared radius of the support.
    :return:
    """
    orig_shape = sq_d.shape
    sq_d = tf.reshape(sq_d, [-1])
    tf.assert_rank(sq_d, 1)

    idxs = tf.squeeze(tf.where(sq_d < sq_r), axis=-1)
    sq_d1 = tf.gather(sq_d, indices=idxs)
    r = tf.sqrt(sq_d1 / sq_r)     # Normalized radius
    r = tf.pow(1 - r, 4)*(4*r + 1)
    ret = tf.tensor_scatter_nd_add(tf.zeros_like(sq_d), indices=tf.expand_dims(idxs, -1), updates=r)
    return tf.reshape(ret, shape=orig_shape)


def crbf_wendland_d3c2_factory(peak: float, spread: float) -> RBF:
    """
    Get a Wendland compactly supported RBF factory. The returned function is expecting a tensor containing the
    squared distances as argument.
    :param peak:
    :param spread: The spread parameter.
    :return: A callable implementing the RBF.
    """
    def fun(sq_distances: tf.Tensor) -> tf.Tensor:
        return crbf_wendland_d3c2(sq_d=sq_distances, sq_r=tf.square(spread)) * tf.convert_to_tensor(peak)
    return fun


def frac_part(x: tf.Tensor) -> tf.Tensor:
    return x - tf.floor(tf.math.abs(x))*tf.math.sign(x)


def random_masks_like(values: Union[tf.Tensor, Sequence[tf.Tensor]], p: float) -> Union[tf.Tensor, List[tf.Tensor]]:
    if isinstance(values, (tf.Tensor, tf.Variable)):
        return tf.random.uniform(shape=values.shape) < p
    return list(map(lambda m: tf.random.uniform(shape=m.shape) < p, values))


def bezier(ctrl_points: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    """
    Compute multiple points (at multiple ts) of a Bezier curve. The degree of the curve is inferred directly from
    tensor's shape.
    :param ctrl_points: Control points tensor, the expected shape is: [ctrl_point_count, dim]
    :param t: t tensor of either shape [t_0, t_1, ...] or scalar
    :return: Tensor containing the curve points, the expected shape is [t_count, dim]
    """
    with tf.name_scope('bezier'):
        t = tf.expand_dims(tf.reshape(t, shape=[-1]), axis=-1)  # t form: [[t_1], [t_2], [t_3], ...]

        ret = tf.zeros(shape=[t.shape[0], ctrl_points.shape[1]])
        n = ctrl_points.shape[0] - 1
        for k in range(n+1):
            ret += tf.cast(tf_binom(n, k), dtype=TF_FLOAT)*tf.pow(t, k)*tf.pow(1. - t, n - k)*ctrl_points[k]
        return ret


def bezier_multi(ctrl_points: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    """
    Compute multiple points (at multiple ts) for multiple Beziers. Each Bezier is identified by a tuple of control
    points. The degree of the Bezier is the same for all curves and it is inferred directly from tensor's shape.
    :param ctrl_points: Control points tensor of the form [curve_count, ctrl_point_count, dim]
    :param t: t tensor of either form [t_0, t_1, ...] or scalar
    :return: Tensor containing the curve points, the expected shape is [curve_count, t_count, dim]
    """
    with tf.name_scope('bezier_multi'):
        # t_orig_rank = tf.rank(t)
        t = tf.reshape(t, shape=[-1])  # t form: [t_0, t_1, t_2, ...]
        einsum_pattern = 'i,jk->jik'

        ret = tf.zeros(shape=[ctrl_points.shape[0], t.shape[0], ctrl_points.shape[2]])
        n = ctrl_points.shape[1] - 1    # degree
        # TODO create and use binom_all
        for k in range(n + 1):
            ret += tf.cast(tf_binom(n, k), dtype=TF_FLOAT)*tf.einsum(einsum_pattern,
                                                                     tf.pow(t, k)*tf.pow(1. - t, n - k),
                                                                     ctrl_points[:, k, :])
        # if t_orig_rank == 0:
        #    return tf.squeeze(ret, axis=1)
        return ret


def path_angles(paths: tf.Tensor) -> tf.Tensor:
    with tf.name_scope('path_angles'):
        if paths.shape[1] <= 1:     # 1 or 0 points considered angle 0.
            return tf.zeros(shape=paths.shape[0], dtype=TF_FLOAT)

        ret = tf.square(paths[:, 1:, :] - paths[:, 0:-1, :])
        ret = tf.atan2(ret[:, :, 0], ret[:, :, 1])
        return ret


def mutual_sq_distances(points):
    """
    Calculates the mutual distances among all points in the given tensor. The expected shape of the input tensor
    is [point_idx, dim].
    :param points:
    :return:
    """
    tf.assert_rank(points, 2)
    if len(points) <= 1:
        return tf.convert_to_tensor([], dtype=points.dtype)
    d = [tf.reduce_sum(tf.square(points[(k + 1):] - points[k]), axis=-1) for k in range(len(points)-1)]
    return tf.concat(d, axis=0)


def path_length(paths: tf.Tensor) -> tf.Tensor:
    """
    Measure the length (Euclidean distance) of linear piecewise paths. The expected shape of the input tensor
    is [curve_count, t_count, dim].
    :param paths:
    :return:
    """
    with tf.name_scope('path_length'):
        if paths.shape[1] <= 1:     # 1 or 0 points => zero distance
            return tf.zeros(shape=paths.shape[0], dtype=TF_FLOAT)

        lenghts = tf.square(paths[:, 1:, :] - paths[:, 0:-1, :])
        lenghts = tf.sqrt(tf.reduce_sum(lenghts, axis=-1))
        lenghts = tf.reduce_sum(lenghts, axis=-1)   # Sum-up all segment lengths
        return lenghts


def bounding_box(points: tf.Tensor) -> tf.Tensor:
    points = tf.transpose(points)
    return tf.stack([
        tf.reshape(tf.reduce_min(points, axis=1), shape=[-1]),
        tf.reshape(tf.reduce_max(points, axis=1), shape=[-1])])


def gauss_rbf(sq_d, spread):    # TODO remove, substitute with gauss_rbf_factory
    return tf.exp(-sq_d / (2.0 * tf.square(spread)))


def gauss_rbf_factory(peak: float, spread: float) -> RBF:
    """
    Get a Gaussian RBF factory. The returned function is expecting a tensor containing the squared distances as
    argument.
    :param peak:
    :param spread: The spread parameter.
    :return: A callable implementing the RBF.
    """
    def fun(sq_distances: tf.Tensor):
        return tf.exp(-sq_distances / (2.0 * tf.square(spread))) * peak
    return fun


_rbf_factories = {
    'gauss': gauss_rbf_factory,
    'wendland_d3c2': crbf_wendland_d3c2_factory,
    'wendland': crbf_wendland_d3c2_factory,
}


def rbf_factory(name: str, peak: float, spread: float, **kwargs) -> RBF:
    """
    Create a RBF given name, peak, spread an optional parameters.
    :param name: The name of the RBF (eg. 'gauss')
    :param peak: The peak of the RBF function, that is the value at distance 0.
    :param spread: The spread of the RBF, this is specific to each function type.
    :param kwargs:
    :return: A RBF.
    """
    factory = _rbf_factories[name]
    return factory(peak=peak, spread=spread, **kwargs)


def random_parametric_steps(steps):
    tf.debugging.assert_greater_equal(steps, 2)
    t = tf.sort(tf.random.uniform(shape=[steps - 2], dtype=TF_FLOAT))
    return tf.concat([[0.], t, [1.]], axis=0)


def random_undirected_adjacency(k: Union[int, tf.Tensor], *, loops=True, reflect=True) -> tf.Tensor:
    """
    Create a random adjacency matrix for an undirected graph.
    :param k: The number of vertexes
    :param loops:
    :param reflect:
    :return:
    """
    k = tf.convert_to_tensor(k, dtype=tf.int64)
    adjmat = np.random.uniform(size=(k, k)) * tf.cast(tri_ones_lower(k), dtype=TF_FLOAT)
    if loops:
        adjmat = adjmat * (1. - tf.eye(k, dtype=TF_FLOAT))
    if reflect:
        adjmat += tf.transpose(adjmat)
        tf.assert_equal(adjmat, tf.transpose(adjmat))
    return adjmat


def random_uniform_disk(size) -> tf.Tensor:
    """
    Random sampling over the unit disk.
    Ref http://mathworld.wolfram.com/DiskPointPicking.html
    :param size:
    :return:
    """
    r = tf.sqrt(tf.random.uniform(shape=(size, ), dtype=TF_FLOAT))
    t = tf.random.uniform(shape=(size, ), dtype=TF_FLOAT) * 2.*np.pi
    return tf.stack([r*tf.sin(t), r*tf.cos(t)], axis=1)


def random_uniform_hypersphere(size, dim) -> tf.Tensor:
    """
    Generate random points uniformly distributed in a hyper-sphere.
    :param size: The number of points to be generated.
    :param dim: The dimension of the hyper-sphere.
    :return: A tensor, expected shape: [size, dim].
    """
    if dim == 1:
        return tf.random.uniform(shape=(size, ), dtype=TF_FLOAT) * 2.0 - 1.
    if dim == 2:
        return random_uniform_disk(size)
    if dim == 3:
        # TODO implement: http://mathworld.wolfram.com/SpherePointPicking.html
        warnings.warn('Sphere point picking is not implemented properly')
        return tf.random.uniform(shape=(size, dim), dtype=TF_FLOAT) * 2.0 - 1.
    raise ValueError(f'Unsupported space dimension: {dim}')


def piecewise_linear_curve_closest_pts(curve_pts, centers) -> tf.Tensor:
    """
    Compute the closest distances between a set of centers and a piecewise linear curve.
    :param curve_pts: A set of points representing the way points of the curve. Expected shape: [pt_count, dim]
    :param centers: A set of center points, expected shape: [center_count, dim]
    :return: A list of squared distances, expexted shape: [center_count. pt_count - 1]
    """
    # curve_pts.shape=(pt_count, dim)
    # centers.shape=(center_count, dim)
    with tf.name_scope('piecewise_linear_curve_closest_pts'):
        p0s = curve_pts[:-1]    # Segment starting points
        seg_dirs = curve_pts[1:] - curve_pts[:-1]      # Segment directions, seg_dirs.shape=(pt_count-1, dim)
        sq_lengths = tf.reduce_sum(tf.square(seg_dirs), axis=-1)  # sq_lengths.shape=(pt_count-1, )
        centers_p0s_dirs = tf.expand_dims(centers, axis=1) - p0s  # centers_p0s_dirs.shape=(center_count, pt_count-1, dim)

        ts = tf.einsum('ijk,jk -> ij', centers_p0s_dirs, seg_dirs)   # ts.shape=(center_count, pt_count-1)
        ts = ts / sq_lengths
        ts = tf.maximum(0., tf.minimum(1., ts))

        ps = tf.einsum('ij,jk -> ijk', ts, seg_dirs) + p0s     # ps.shape=(center_count, pt_count-1, dim)
        # ps = tf.reshape(ps, [-1, ps.shape[-1]])                # ps.shape=(center_count * (pt_count-1), dim)

        sq_distances = tf.reduce_sum(tf.square(ps - tf.expand_dims(centers, axis=1)), axis=-1)
        # sq_distances.shape=(center_count, pt_count-1)

        return sq_distances

