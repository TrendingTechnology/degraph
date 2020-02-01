from .types import StaticShape, RBF, Adjacency
from .math import tf_hgrid_coords, hgrid_normalize, bezier_multi, tri_ones_lower, tf_repeat_1d
from .math import piecewise_linear_curve_closest_pts
from .math import random_uniform_hypersphere, mutual_sq_distances, gauss_rbf, rbf_factory, crbf_wendland_d3c2
from .config import TF_FLOAT, EPSILON, BEZIER_DEFAULT_DEGREE
from .model import create_entity_name, get_active_model, Entity, SingleVarContrib, TensorRef, expand_tensor_ref
from .model import stop_gradient_tape
from .utils import join_ident, export, apply_colormap

from typing import Sequence, List, Optional, Union, Dict, Callable, Tuple
import abc
import tensorflow as tf
import pandas as pd
import numpy as np
import h5py
from copy import copy
from contextlib import nullcontext


RBF_DEFAULT = 'gauss'


class Lambda(Entity):
    """
    Entity wrapping an arbitrary expression involving tensors.
    """

    def __init__(self, fun: Callable, tensor_ref: Union[TensorRef, Dict[str, TensorRef], List[TensorRef]], *,
                 name=None, enable_grad: bool = True):
        """
        Init lambda entity.
        :param fun: The callable to be wrapped.
        :param tensor_ref: Tensor ref or a list or dict of tensor refs.
        :param name: The name of the entity.
        :param enable_grad: Boolean (default true) specifying whether the gradient tape should be enabled when in
        training mode.
        """
        super().__init__(name=name)
        self.fun = fun
        self.tensor_ref = copy(tensor_ref)
        self.enable_grad = enable_grad
        self.add_to_active_model()

    def get_value(self):
        tensor_ref = self.tensor_ref
        if isinstance(tensor_ref, dict):
            params = map(lambda it: (it[0], expand_tensor_ref(it[1])), tensor_ref.items())
            params = dict(params)

            with stop_gradient_tape(not self.enable_grad):
                return self.fun(**params)

        if isinstance(tensor_ref, list):
            params = map(lambda it: expand_tensor_ref(it), tensor_ref)
            with stop_gradient_tape(not self.enable_grad):
                return self.fun(*params)

        param = expand_tensor_ref(tensor_ref)
        with stop_gradient_tape(not self.enable_grad):
            return self.fun(param)

    def _var_name(self) -> str:
        return join_ident(self.name, 'value')

    def get_value_ref(self) -> TensorRef:
        def fun():
            model = get_active_model()
            assert model
            return model.runtime_context[self._var_name()]
        return fun

    def run(self):
        get_active_model().runtime_context[self._var_name()] = self.get_value()


@export
def lambda_(fun: Callable, tensor_ref: Union[TensorRef, Dict[str, TensorRef], List[TensorRef]], *,
            name=None, enable_grad: bool = True):
    """
    Create an entity that wraps an arbitrary expression involving tensors.
    :param fun: A lambda function invoked with the expanded tensor refs declared in the parameter tensor_ref.
    :param tensor_ref: A tensor ref, a list of tensor refs or a dictionary. In the case of a dictionary the function
    will be invoked with named parameters whereas in the other cases positional parameters will be used.
    :param name: Name of the entity wrapping the expression.
    :param enable_grad: if true (default) the tensor tape is enabled while in training mode.
    :return: An entity wrapping the expression implemented in fun.
    """
    return Lambda(fun=fun, tensor_ref=tensor_ref, name=name, enable_grad=enable_grad).get_value_ref()


TensorRefs = Union[Sequence[TensorRef], TensorRef]


def tensor_refs_clone(refs: TensorRefs) -> TensorRefs:
    """
    Clone a tensor ref or a sequence of tensor refs.
    :param refs:
    :return:
    """
    if isinstance(refs, (tf.Tensor, tf.Variable, Callable)):
        return refs
    return list(refs)


@export
def reduce_sum(tensor_refs: Union[Sequence[TensorRef], TensorRef], *, name=None):     # TODO find a more proper name
    tensor_refs = tensor_refs_clone(tensor_refs)

    def fun(*tensors):
        return tf.reduce_sum(tf.convert_to_tensor(tensors))
    return lambda_(fun, tensor_ref=tensor_refs, name=name)


@export
def reduce_max(tensor_refs: Union[Sequence[TensorRef], TensorRef], *, name=None):     # TODO find a more proper name
    tensor_refs = tensor_refs_clone(tensor_refs)

    def fun(*tensors):
        return tf.reduce_max(tf.convert_to_tensor(tensors))
    return lambda_(fun, tensor_ref=tensor_refs, name=name)


@export
def aggregate_rasters(rasters: Union[TensorRef, List[TensorRef]], *, bias: float = 0., name=None):
    """
    Aggregate a list of images. The activation function ReLU is applied after the input tensors are stacked and added.
    :param rasters: A list of tensor refs representing the input images.
    :param bias: Bias value.
    :param name: Name of the entity.
    :return: An entity implementing the aggregation.
    """
    def adder(*values):
        values = list(filter(lambda val: val is not None, values))
        ret = tf.reduce_sum(tf.stack(values), axis=0) + tf.convert_to_tensor(bias, dtype=TF_FLOAT)
        ret = tf.nn.relu(ret)
        # ret = tf.math.exp(ret) - 1  # TODO param for lambda here
        # TODO create debug mode, in that case summarise some statistics for the raster
        return ret
    return lambda_(adder, tensor_ref=rasters, name=name)


class RBFNet:
    """
    Radial basis function network.
    """

    def __init__(self, rbf: RBF):
        """
        Init RBF network.
        :param rbf: The radial basis function, eg the value returned by degraph.math.rbf_factory(...).
        """
        self.rbf = rbf

    @staticmethod
    def create_grid(shape: StaticShape):
        return tf.cast(tf_hgrid_coords(shape), dtype=TF_FLOAT)

    @tf.function
    def __call__(self, shape: StaticShape, centres: tf.Tensor):
        grid = tf.cast(tf.expand_dims(tf_hgrid_coords(shape), axis=1), dtype=TF_FLOAT)
        grid = tf.reduce_sum(tf.square(grid - centres), axis=-1)
        grid = self.rbf(grid)
        grid = tf.reduce_sum(grid, axis=-1)  # form a mixture
        grid = tf.reshape(grid, shape=shape)  # reshape values to an image
        return grid

    @tf.function
    def grid_lines_on_grid(self, shape: StaticShape, centres: tf.Tensor):
        """
        Apply radial basis function on a grid using distances from lines aligned with the axes.
        """
        grid_acc = tf.zeros(shape=tf.reduce_prod(shape), dtype=TF_FLOAT)
        grid = tf.cast(tf.expand_dims(tf_hgrid_coords(shape), axis=1), dtype=TF_FLOAT)

        for k in range(centres.shape[1]):
            level = tf.squeeze(tf.square(grid[:, :, k:k+1] - centres[:, k:k+1]), axis=-1)
            level = self.rbf(level)

            level = tf.reduce_sum(level, axis=-1)  # form a mixture
            grid_acc += level

        grid_acc = tf.reshape(grid_acc, shape=shape)  # reshape values to an image
        return grid_acc


class RBFNetEntityBase(Entity):
    def __init__(self, peak: float = 1.0, spread: float = 1.0, name=None):
        super().__init__(name=name)
        self.peak = peak
        self.spread = spread
        self._rbf_net = None

    def update_rbf_net(self, force: bool = True):
        if not force and self._rbf_net is not None:
            return

        peak = tf.convert_to_tensor(self.peak, dtype=TF_FLOAT)
        assert tf.rank(peak) == 0
        spread = tf.convert_to_tensor(self.spread, dtype=TF_FLOAT)
        assert tf.rank(spread) == 0
        # TODO use Normal core, use partial also try truncated normal
        self._rbf_net = RBFNet(rbf=lambda values: tf.exp(-values/(2.0 * tf.square(spread))) * peak)
        # self._rbf_net = RBFNet(rbf=lambda values: crbf_wendland_d3c2(values, sq_r=tf.square(spread)) * peak)

    # TODO two-phases: get_centres and get_layer, in the middle optionally normalize

    @abc.abstractmethod
    def get_centres(self) -> tf.Tensor:
        """
        Returns a tensor containing the normalized coordinates of the Gaussian centres. The coordinates are expected
        to belong to the interval [-1., 1.].
        :return:
        """
        pass

    def get_layer(self, shape: StaticShape) -> tf.Tensor:
        centres = self.get_centres()
        assert len(shape) == centres.shape[-1]  # Assert centres and grid have the same dimension

        # TODO assert centres.shape compatibility with len(shape)
        # Note centres coordinates are assumed in the interval [-1., 1.]
        centres = (centres + 1.)/2. * tf.convert_to_tensor(shape, dtype=TF_FLOAT)     # Scale to layer coordinates size

        self.update_rbf_net(force=False)
        grid = self._rbf_net(shape=shape, centres=centres)
        return grid


@export
class GraphRepr:    # TODO extend tf.Module?
    """
    Symmetric directed graph representation.
    """

    # TODO Another ideas would be of getting rid of this class and have individual entities for var and other op,
    #   this is in line with the idea of using other type of coordinates (eg. polar).

    HDF5_SER_TYPE = 'degraph.GraphRepr'
    DEF_NAME_PREFIX = 'graph_def'

    def __init__(self, adjacency: Adjacency, dim: int = 2, name=None, **kwargs):
        """
        Init the graph representation object.
        :param adjacency: Adjacency matrix, this can be either a Numpy array, a Pandas DataFrame or a tensor.
        :param dim: Number of spatial dimensions of this representation, currently many internal components are limited
        to the 2D case it is the only option.
        :param name:
        :param kwargs:
        """
        # TODO optional param positions_generator (space positions)
        self._name = create_entity_name(name, prefix=self.DEF_NAME_PREFIX)
        self._static = False

        if '_internal_skip_init' in kwargs:
            return
        if len(kwargs) != 0:
            raise ValueError(f'Unsupported extra parameters: {kwargs.keys()}')

        dim = int(dim)
        assert dim >= 1
        if dim not in (2, 3):
            raise ValueError(f'Space dimensions not supported: {dim}')

        if isinstance(adjacency, pd.DataFrame):
            assert adjacency.index == adjacency.columns
            adjacency = np.array(adjacency)
        if isinstance(adjacency, np.ndarray):
            adjacency = tf.convert_to_tensor(adjacency)
        if not isinstance(adjacency, (tf.Tensor, tf.Variable)):
            raise ValueError(f'Unsupported type for adjacency matrix: {type(adjacency)}')

        assert len(adjacency.shape) == 2 and adjacency.shape[0] == adjacency.shape[1]
        # Mask upper triangular values of the adjacency matrix
        adjacency = tf.cast(adjacency, dtype=TF_FLOAT) * tf.cast(tri_ones_lower(adjacency.shape[0]), dtype=TF_FLOAT)
        adjacency /= tf.reduce_max(tf.abs(adjacency)) + EPSILON      # Normalize edge abs weights
        edge_count = tf.math.count_nonzero(adjacency)
        edge_extreme_indexes = tf.where(adjacency)
        assert edge_count == edge_extreme_indexes.shape[0] and edge_extreme_indexes.shape[1] == 2

        self.adjacency = adjacency
        # The indexes of the extreme points of each edge. The indexes point at the first axis of the variable positions.
        self.edge_extreme_indexes = edge_extreme_indexes

        with tf.name_scope(self.name):
            self.positions = tf.Variable(random_uniform_hypersphere(size=adjacency.shape[0], dim=dim), trainable=True,
                                         name='positions')

            total_ctrl_point_count = edge_extreme_indexes.shape[0] * ((BEZIER_DEFAULT_DEGREE+1)-2)
            ctrl_points_init = tf.reshape(random_uniform_hypersphere(size=total_ctrl_point_count, dim=dim),
                                          shape=(-1, (BEZIER_DEFAULT_DEGREE+1)-2, dim))
            self.edge_internal_ctrl_points = tf.Variable(ctrl_points_init, trainable=True,
                                                         name='edge_internal_ctrl_points')

    def empty_positions(self) -> tf.Tensor:
        return tf.zeros(shape=(0, self.dim), dtype=TF_FLOAT)

    @property
    def name(self):
        return self._name

    @property
    def static(self) -> bool:
        """
        Boolean property, true if the current graph representation is static (i.e. the positions are not of the type
        tf.Variable).
        :return:
        """
        return self._static

    def copy(self, *, static=False, name: Optional[str] = None):
        """
        Clone the current instance of GraphRepr with the excpection of the static flag which is passed through the
        arguments.
        :param static: When static is set the variables are transformed to static tensors in the destination object.
        This is useful to take snapshots of the status.
        :param name: Optional name for the graph, otherwise an automatic one is generated.
        :return:
        """
        obj = GraphRepr(np.asarray(0), _internal_skip_init=True)
        obj.adjacency = self.adjacency
        obj.edge_extreme_indexes = self.edge_extreme_indexes
        obj._static = static
        obj._name = create_entity_name(name, prefix=self.DEF_NAME_PREFIX)

        # Convert variables to tensors, note that when GraphRepr.static is set these are already tensors.
        positions = tf.convert_to_tensor(self.positions)
        edge_internal_ctrl_points = tf.convert_to_tensor(self.edge_internal_ctrl_points)
        if static:
            obj.positions = positions
            obj.edge_internal_ctrl_points = edge_internal_ctrl_points
        else:
            obj.positions = tf.Variable(positions, trainable=True)
            obj.edge_internal_ctrl_points = tf.Variable(edge_internal_ctrl_points, trainable=True)
        return obj

    def get_ctrl_points_vars(self) -> List[tf.Variable]:
        """
        Get the control point variables relative to the vertexes and the edges.
        :return: A list of elements of type tf.Variable.
        """
        if self.static:
            return []
        return [self.positions, self.edge_internal_ctrl_points]

    def autoscale_on_vertexes(self, center_scale: float = 1.):
        points = self.get_positions()
        limits = tf.reduce_min(points), tf.reduce_max(points)

        def get_update(value):
            return tf.math.divide_no_nan(value - limits[0],
                                         limits[1] - limits[0]) * center_scale + (1. - center_scale)/2.

        if self.static:
            self.positions = get_update(self.positions)
            self.edge_internal_ctrl_points = get_update(self.edge_internal_ctrl_points)
        else:
            for var in self.get_ctrl_points_vars():
                var.assign(get_update(var.value()))
        return self

    def serialize(self, fobj):
        """
        Serialize the current object in HDF5 format using the file-like object provided.
        :param fobj: A file-like object
        :return:
        """
        with h5py.File(fobj, mode='w') as f:
            f['type'] = self.HDF5_SER_TYPE
            f['adjacency'] = self.adjacency.numpy()
            f['edge_extreme_indexes'] = self.edge_extreme_indexes.numpy()
            f['positions'] = self.positions.numpy()
            f['edge_internal_ctrl_points'] = self.edge_internal_ctrl_points.numpy()

    @property
    def dim(self) -> int:
        """
        The number of spatial dimentions of this representation.
        :return:
        """
        return self.positions.shape[1]

    def get_positions(self) -> tf.Tensor:
        """
        Get a tensor containing the positions of the vertexes. The expected shape is [pt_count, dim].
        :return:
        """
        # TODO optionally we may include a calculation here, eg polar coordinates to cartesian
        return tf.convert_to_tensor(self.positions)

    def get_positions_ref(self) -> TensorRef:
        """
        Get a tensor ref relative to the positions of the vertexes. See get_positions().
        :return:
        """
        def fun() -> tf.Tensor:
            return self.get_positions()
        return fun

    def get_edges_ctrl_points(self) -> tf.Tensor:
        """
        Get the control points of the edges, the shape of the tensor is [edge_count, ctrl_point_count, dim]
        :return:
        """
        # edge_extreme_points, shape: [edge_count, 2, dim]
        edge_extreme_points = tf.gather(self.positions, indices=self.edge_extreme_indexes)
        # Compose a tensor with edge's end point positions and internal control points
        return tf.concat([edge_extreme_points[:, 0:1, :],   # start points
                          self.edge_internal_ctrl_points,   # internal points
                          edge_extreme_points[:, 1:2, :]], axis=1)  # end points

    def get_edges_ctrl_points_ref(self) -> TensorRef:
        def fun() -> tf.Tensor:
            return self.get_edges_ctrl_points()
        return fun


class Vertexes(SingleVarContrib):
    """
    Entity representing the vertexes of a graph.
    """

    def __init__(self, graph: GraphRepr, *, trainable: bool = True, name=None):
        super().__init__(name=name)
        self.graph = graph
        self.trainable = bool(trainable)
        self.add_to_active_model()

    def get_trainable_variables(self) -> Optional[Sequence[tf.Tensor]]:
        return [self.graph.positions] if self.trainable else []

    def get_value(self) -> tf.Tensor:
        return tf.convert_to_tensor(self.graph.positions)


@export
def vertexes(graph: GraphRepr, *, trainable: bool = True, name=None) -> TensorRef:
    """
    Create an entity that represents the vertexes of a graph.
    :param graph: The graph object.
    :param trainable: If true the variable relative to the positions of the vertexes are marked as trainable.
    :param name: The name of the entity.
    :return: The entity object.
    """
    return Vertexes(graph=graph, trainable=trainable, name=name).get_value_ref()


class PiecewiseLinearEdges(SingleVarContrib):
    def __init__(self, graph: GraphRepr, *, trainable: bool = True, steps: int = 25, space_radius: float = 10.,
                 name=None):
        super().__init__(name=name)
        trainable = bool(trainable)
        self.graph = graph
        self.trainable = trainable
        self.steps = steps
        self.space_radius = space_radius

        self.last_pts = graph.empty_positions()
        self.add_to_active_model()

    def get_trainable_variables(self) -> Optional[Sequence[tf.Tensor]]:
        return [self.graph.edge_internal_ctrl_points] if self.trainable else []

    @staticmethod
    @tf.function
    def _bezier(ctrl_points, t):
        return bezier_multi(ctrl_points=ctrl_points, t=t)

    def get_value(self) -> tf.Tensor:
        t = tf.linspace(0., 1., self.steps)   # TODO selector between stochastic and regular
        # Note points too close can create problems to the gradient...
        # t = random_parametric_steps(self.steps)

        # TODO use get_edges_ctrl_points_ref() and activate tf.function
        # return bezier_multi(ctrl_points=self.graph.get_edges_ctrl_points(), t=t)
        return self._bezier(ctrl_points=self.graph.get_edges_ctrl_points(), t=t)


@export
def piecewise_linear_edges(graph: GraphRepr, *, trainable: bool = True, steps: int = 25, space_radius: float = 10.,
                           name=None) -> TensorRef:
    obj = PiecewiseLinearEdges(graph, trainable=trainable, steps=steps, space_radius=space_radius, name=name)
    return obj.get_value_ref()


@export
def unit_sphere_bounds_loss(points_tensor_ref: TensorRef, *, factor: float = 1.0):
    """
    Get a loss that penalises points laying outside the unit hyper-sphere centred in the origin.
    :param points_tensor_ref: A tensor ref containing the coordinates of the points.
    :param factor: A multiplicative factor for the loss.
    :return: An entity implementing the loss.
    """
    @tf.function
    def fun(points: tf.Tensor):
        if factor <= 0.:
            return tf.convert_to_tensor(0., dtype=TF_FLOAT)
        # TODO assert rank, support shape [..., dim]
        return factor * tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.square(points), axis=-1) - 1.0))
    return lambda_(fun, points_tensor_ref)


class VertexDistancesLoss(SingleVarContrib):    # TODO remove, create simple func with lambda
    def __init__(self, points_tensor_ref: TensorRef, *, factor: float = 1.0, spread: float = 1.0):
        super().__init__()
        self.points_tensor_ref = points_tensor_ref
        self.factor = factor
        self.spread = spread
        self.add_to_active_model()

    def get_value(self):
        if self.factor <= 0.:
            return 0.

        model = get_active_model()
        assert model is not None
        pts = expand_tensor_ref(self.points_tensor_ref)  # TODO assert rank

        # TODO set tf.function here!
        # Scale to layer coordinates size
        positions = tf.add(pts, 1.) / 2. * tf.convert_to_tensor(model.shape, dtype=TF_FLOAT)
        distances = mutual_sq_distances(positions)
        distances = tf.reduce_sum(gauss_rbf(sq_d=distances, spread=self.spread))
        return distances * self.factor


@export
def mse_loss(value_ref: TensorRef, *, factor: float = 1.0) -> TensorRef:
    """
    Get a MSE Loss.
    :param value_ref: The input tensor.
    :param factor: A multiplicative factor for the loss.
    :return: An entity implementing the loss.
    """
    @tf.function
    def fun(value: tf.Tensor):
        if factor <= 0.:
            return tf.convert_to_tensor(0., dtype=TF_FLOAT)

        value = tf.reshape(value, shape=(-1, ))
        # TODO Optional param mask, basically it should mask the values that are part of the mean.
        # return tf.reduce_mean(tf.square(value)) * factor
        return tf.reduce_mean(tf.square(value * tf.sqrt(factor)))

    return lambda_(fun, value_ref)


@export
def sse_loss(value_ref: TensorRef, *, factor: float = 1.0) -> TensorRef:
    """
    Get a sum of squares loss.
    :param value_ref: The input tensor.
    :param factor: A multiplicative factor for the loss.
    :return: An entity implementing the loss.
    """
    @tf.function
    def fun(value: tf.Tensor):
        if factor <= 0.:
            return tf.convert_to_tensor(0., dtype=TF_FLOAT)

        value = tf.reshape(value, shape=(-1, ))
        return tf.reduce_sum(tf.square(value)) * factor

    return lambda_(fun, value_ref)


class RBFNetRaster(SingleVarContrib):
    """
    An entity that creates a raster using a radial basis function network.
    """

    def __init__(self, points_tensor_ref: TensorRef, *, shape: StaticShape,
                 rbf: str = RBF_DEFAULT, peak: float = 1.0, spread: float = 1.0, name=None):
        """
        Init the entity.
        :param points_tensor_ref: A tensor ref referencing a tensor with expected shape: [point_count, dim].
        The points are used as centres of the radial basis functions.
        :param shape: The shape of the raster.
        :param rbf: The rbf to be used, see degraph.math.rbf_factory.
        :param peak: The peak of the RBF.
        :param spread: The spread of the RBF.
        :param name: The name of the entity.
        """
        super().__init__(name=name)
        self.points_tensor_ref = points_tensor_ref
        self.shape = shape
        self.rbf = rbf
        self.peak = peak
        self.spread = spread
        self._rbf_net = None
        self.add_to_active_model()

    def update_rbf_net(self, force: bool = True):
        if not force and self._rbf_net is not None:
            return

        peak = tf.convert_to_tensor(self.peak, dtype=TF_FLOAT)
        assert tf.rank(peak) == 0
        spread = tf.convert_to_tensor(self.spread, dtype=TF_FLOAT)
        assert tf.rank(spread) == 0

        self._rbf_net = RBFNet(rbf=rbf_factory(self.rbf, peak=peak, spread=spread))

    def get_raster(self) -> tf.Tensor:
        centres = expand_tensor_ref(self.points_tensor_ref)  # TODO assert rank
        shape = self.shape
        assert len(shape) == centres.shape[-1]  # Assert centres and grid have the same dimension

        # TODO assert centres.shape compatibility
        # Note centres coordinates are assumed in the interval [-1., 1.]
        centres = (centres + 1.)/2. * tf.convert_to_tensor(shape, dtype=TF_FLOAT)     # Scale to layer coordinates size

        self.update_rbf_net(force=False)
        grid = self._rbf_net(shape=shape, centres=centres)
        return grid

    def get_value(self) -> tf.Tensor:
        return self.get_raster()


@export
def rbf_net_raster(points_tensor_ref: TensorRef, *, shape: StaticShape, rbf: str = RBF_DEFAULT,
                   peak: float = 1.0, spread: float = 1.0):
    obj = RBFNetRaster(points_tensor_ref, rbf=rbf, shape=shape, peak=peak, spread=spread)
    return obj.get_value_ref()


class RBFSegNetRaster(SingleVarContrib):
    def __init__(self, segments_tensor_ref: TensorRef, *, shape: StaticShape,
                 rbf: str = RBF_DEFAULT, peak: float = 1.0, spread: float = 1.0, name=None):
        super().__init__(name=name)
        self.segments_tensor_ref = segments_tensor_ref
        self.shape = shape
        self.rbf = rbf
        self.peak = peak
        self.subsample = 1.0
        self.spread = spread
        self._rbf_net = None
        self.add_to_active_model()

    def update_rbf_net(self, force: bool = True):
        if not force and self._rbf_net is not None:
            return

        peak = tf.convert_to_tensor(self.peak, dtype=TF_FLOAT)
        assert tf.rank(peak) == 0
        spread = tf.convert_to_tensor(self.spread, dtype=TF_FLOAT)
        assert tf.rank(spread) == 0

        self._rbf_net = RBFNet(rbf=rbf_factory(self.rbf, peak=peak, spread=spread))

    def get_raster(self) -> tf.Tensor:
        segments = expand_tensor_ref(self.segments_tensor_ref)  # TODO assert rank
        shape = self.shape
        assert len(shape) == segments.shape[-1]  # Assert centres and grid have the same dimension

        # Note centres coordinates are assumed in the interval [-1., 1.]
        segments = (segments + 1.)/2. * tf.convert_to_tensor(shape, dtype=TF_FLOAT)  # Scale to layer coordinates size
        self.update_rbf_net(force=False)
        grid_coords = self._rbf_net.create_grid(shape=shape)

        subsample = self.subsample
        if subsample != 1.0:
            with stop_gradient_tape():
                segments = tf.random.shuffle(segments)[:int(subsample*len(segments))]

        # TODO tf.function
        raster = tf.zeros(shape=shape, dtype=TF_FLOAT)
        for segment in segments:
            layer = piecewise_linear_curve_closest_pts(curve_pts=segment, centers=grid_coords)
            layer = self._rbf_net.rbf(layer)
            layer = tf.reduce_sum(layer, axis=-1)
            raster += tf.reshape(layer, shape=shape)
        return raster

    def get_value(self) -> tf.Tensor:
        return self.get_raster()


@export
def rbf_segnet_raster(points_tensor_ref: TensorRef, *, shape: StaticShape, peak: float = 1.0, spread: float = 1.0):
    obj = RBFSegNetRaster(points_tensor_ref, shape=shape, peak=peak, spread=spread)
    return obj.get_value_ref()


def _scope_prepare(scope: str) -> Tuple[object, str]:
    """
    Parse a scope string a return a tuple consisting of context manager for the assignation of the tf's scope
    and a string representing the summary name. The scope is of the form "<ident1>.<ident2>. ... .<ident3>", the
    righmost identifier is used as summary name whereas the prefix is used as scope name.
    :param scope: A string containing a qualified name.
    :return:
    """
    splits = scope.rsplit('.', 1)
    if any(map(lambda v: len(v) == 0, splits)):
        raise ValueError(f'Invalid scope name: {scope}')
    if len(splits) == 1:
        return nullcontext(), splits[0]
    return tf.name_scope(splits[0]), splits[1]


SummaryFunction = Callable[[tf.Tensor, str], None]


class SummaryBase(Entity):
    """
    A base template for summary entities.
    """

    def __init__(self, var: TensorRef, fun: SummaryFunction, *, scope: str, name=None):
        """
        Init summary entity.
        :param var: The variable to the summarised.
        :param fun: A callable of the form fun(tensor, name) that invokes the low level Tensorflow functions.
        :param scope:
        :param name: The name of this entity, note that the name of the summary is taken from parameter scope.
        """
        super().__init__(name=name)
        self.var = var
        self.fun = fun
        self.scope = scope
        self.add_to_active_model()

    def run(self):
        model = get_active_model()
        if model is None:
            return

        value = expand_tensor_ref(self.var)
        scope_ctx, name = _scope_prepare(self.scope)
        with stop_gradient_tape():
            with scope_ctx:
                self.fun(value, name)


@export
def summary_histogram(var: TensorRef, *, scope: str, name=None):
    return SummaryBase(var, fun=lambda value, name_: tf.summary.histogram(name=name_, data=value),
                       scope=scope, name=name)


@export
def summary_scalar(var: TensorRef, *, scope: str, name=None):
    return SummaryBase(var, fun=lambda value, name_: tf.summary.scalar(name=name_, data=value),
                       scope=scope, name=name)


@export
def summary_image(var: TensorRef, *, scope: str, name=None, **kwargs):
    """
    Create an image summary entity. This function wraps tf.summary.image.
    :param var: The tensor to be interpreted as image.
    :param scope:
    :param name: The name of the entity representing this operation, note that the identifier of the summary in
    Tensorboard is determined by the parameter scope.
    :param kwargs: Additional parameters to be passed to tf.summary.image.
    :return:
    """
    def fun(value, name_):
        # Note the tape is not recording here (see SummaryBase.run)
        value = apply_colormap(value)
        if len(value.shape) == 3:
            value = tf.expand_dims(value, axis=0)
        assert len(value.shape) == 4
        tf.summary.image(name=name_, data=value, **kwargs),

    return SummaryBase(var, fun=fun, scope=scope, name=name)
