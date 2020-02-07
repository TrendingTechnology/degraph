"""
Generic plotting utils.
"""

from .entity import GraphRepr, expand_tensor_ref
from .entity import rbf_net_raster, piecewise_linear_edges, aggregate_rasters, vertexes, rbf_segnet_raster
from .model import Model, stop_gradient_tape
from .utils import apply_colormap

from typing import Tuple, Union
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_gauss_rbf(graph: GraphRepr, *, raster_shape: Tuple[int, int],
                   edge_steps: int = 25, vertex_spread: float = 3.,
                   edge_spread: float = 0.5, colormap='jet') -> tf.Tensor:
    """
    Plot a graph using a Gauss RBF network. Note that this operation is not differentiable.
    :param graph: The graph to be plotted.
    :param raster_shape: The shape of the output image.
    :param edge_steps: The number of steps for the piecewise linearization of the edges.
    :param vertex_spread: The spread of the RBF used to represent the vertexes.
    :param edge_spread:
    :param colormap: The colormap to be applied to the RBF representation.
    :return: A tensor representing the plotted image.
    """
    with stop_gradient_tape():
        model = Model()
        with model.as_active():
            vx = vertexes(graph, trainable=False, name='vx')
            edges = piecewise_linear_edges(graph, steps=edge_steps, name='ed', trainable=False)
            # TODO autoscale before creating the raster
            raster = aggregate_rasters([
                rbf_net_raster(vx, shape=raster_shape, rbf='gauss', spread=vertex_spread),
                rbf_segnet_raster(edges, shape=raster_shape, spread=edge_spread)
            ], bias=0.)

        raster = model.run(raster)
        raster = apply_colormap(raster, range_mode='saturate', colormap=colormap)
        return raster


def plot_matplt(graph: GraphRepr, *, ax: plt.Axes, edge_steps: int = 25) -> Axes3D:
    """
    Plot a graph (possibly 3D) using Matplotlib.
    :param graph: The graph to be plotted.
    :param ax: Maplotlib Axes[3D].
    :param edge_steps: The number of steps for the piecewise linearization of the edges.
    """
    with stop_gradient_tape():
        model = Model()
        with model.as_active():
            vx = vertexes(graph, trainable=False, name='vx')
            edges = piecewise_linear_edges(graph, steps=edge_steps, name='ed', trainable=False)

            model.run()
            vx = expand_tensor_ref(vx).numpy()
            edges = expand_tensor_ref(edges).numpy()

        if isinstance(ax, Axes3D):
            ax.set_xlim3d(-1., 1.)
            ax.set_ylim3d(-1., 1.)
            ax.set_zlim3d(-1., 1.)
            ax.set_xticks(np.linspace(-1., 1., 5))
            ax.set_yticks(np.linspace(-1., 1., 5))
            ax.set_zticks(np.linspace(-1., 1., 5))
            for edge in edges:
                ax.plot(edge[:, 0], edge[:, 1], edge[:, 2])
            ax.scatter(vx[:, 0], vx[:, 1], vx[:, 2], marker='o')
        else:
            ax.set_xlim(-1., 1.)
            ax.set_ylim(-1., 1.)
            ax.set_xticks(np.linspace(-1., 1., 5))
            ax.set_yticks(np.linspace(-1., 1., 5))
            for edge in edges:
                ax.plot(edge[:, 0], edge[:, 1])
            ax.scatter(vx[:, 0], vx[:, 1], marker='o')
    return ax

