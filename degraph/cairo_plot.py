"""
Plotting primitives based on the library Cairo (https://www.cairographics.org/).
"""

from .entity import GraphRepr
from .model import stop_gradient_tape, get_active_model
from .utils import create_palette
from .config import TF_FLOAT
from .plot import plot_gauss_rbf

from typing import Tuple, Union, Optional
import math
import cairo
import tensorflow as tf

CurveSource = Union[GraphRepr, tf.Tensor]


class CairoPlotter:
    """
    A graph plotter based on the library Cairo.
    """

    SUPPORTED_BEZIER_DEGREE = 3

    def __init__(self):
        self.edge_width = 0.01/3.
        self.vertex_radius = 0.02
        self.palette = create_palette()

    def set_vertex_config(self, context: cairo.Context):
        context.set_source_rgba(0.8, 0., 0.)

    def set_edge_config(self, context: cairo.Context, index: int):
        context.set_line_width(self.edge_width)
        pal = self.palette
        context.set_source_rgb(*pal[index % pal.shape[0]])
        # context.set_source_rgba(0., 0., 0.)

    def prepare_background(self, context: cairo.Context):
        context.move_to(0., 0.)
        context.rectangle(0., 0., 1., 1.)
        context.set_source_rgba(0.125, 0.125, 0.125)
        context.fill()

    def apply_transform(self, points: tf.Tensor) -> tf.Tensor:
        """
        Apply transform to curve points. This method is meant to be overridden when another transform is required.
        The default transform gets the square with vertexes (-1, -1), (1, 1) into the square with vertexes
        (0, 0), (1, 1).
        :param points: A tensor.
        :return: The transformed tensor.
        """
        return (points + tf.convert_to_tensor(1., dtype=TF_FLOAT)) * 0.5

    def __call__(self, graph: CurveSource, surface: cairo.Surface, shape: Tuple[int, int]):
        """
        Plot a graph on the cairo surface provided.
        :param graph: The graph to be plotted.
        :param surface: The destination surface.
        :param shape: The shape of the space.
        :return:
        """

        if isinstance(graph, GraphRepr):
            ctrl_points = graph.get_edges_ctrl_points()
        elif isinstance(graph, tf.Tensor):
            ctrl_points = graph
        else:
            raise ValueError(f'Unsupported graph type: {type(graph)}')

        def pt_xy(pts, idx):
            return float(pts[idx, 1]), float(pts[idx, 0])

        context = cairo.Context(surface)
        context.scale(shape[1], shape[0])
        self.prepare_background(context)

        if tf.rank(ctrl_points) != 3:
            raise ValueError(f'Invalid ctrl points rank: {tf.rank(ctrl_points)}')
        if ctrl_points.shape[1] != (self.SUPPORTED_BEZIER_DEGREE + 1):
            raise ValueError(f'Unsupported bezier degree: {ctrl_points.shape[1] - 1}')
        if ctrl_points.shape[-1] != 2:
            raise ValueError(f'Unsupported space dim: {ctrl_points.shape[-1]}')

        # TODO enable:
        # ctrl_points = self.apply_transform(ctrl_points)

        for idx, pts in enumerate(ctrl_points):
            self.set_edge_config(context, idx)
            context.move_to(*pt_xy(pts, 0))
            context.curve_to(*[*pt_xy(pts, 1), *pt_xy(pts, 2), *pt_xy(pts, 3)])
            context.stroke()

        self.set_vertex_config(context)
        radius = float(self.vertex_radius)
        extremes = tf.concat([ctrl_points[:, 0, :], ctrl_points[:, -1, :]], axis=0)
        # TODO remove duplicates (use tf.unique on each dim and create a signature and use set())
        for pt in extremes:
            context.arc(pt[1], pt[0], radius, 0, 2*math.pi)
            context.fill()


def plot_svg(fobj, *, graph: CurveSource, shape: Tuple[int, int] = (500, 500)):
    plotter = CairoPlotter()
    with cairo.SVGSurface(fobj, shape[1], shape[0]) as surface:
        plotter(graph=graph, surface=surface, shape=shape)


def plot_png(fobj, *, graph: CurveSource, shape: Tuple[int, int] = (500, 500)):
    plotter = CairoPlotter()
    with cairo.ImageSurface(cairo.Format.ARGB32, shape[1], shape[0]) as surface:
        plotter(graph=graph, surface=surface, shape=shape)
        surface.write_to_png(fobj)


class StatusPlotter:
    """
    A helper class to plot the status of the graph during the training.
    """

    def __init__(self, path_prefix: str, graph: Optional[GraphRepr], raster_shape=(256, 256)):
        """
        Init status plotter object.
        :param path_prefix: Path of the output files comprising the file prefix.
        :param graph: The graph object to be plotted.
        :param raster_shape: The shape of the raster (image).
        """
        self.path_prefix = path_prefix
        self.graph = graph
        self.raster_shape = raster_shape

    def __call__(self, graph: Optional[GraphRepr] = None, index=None,
                 raster_shape: Optional[Tuple[int, int]] = (256, 256), plot_graph=True):
        graph = graph or self.graph
        if graph is None:
            raise ValueError('No graph provided')

        with stop_gradient_tape():
            graph = graph.copy(static=True).autoscale_on_vertexes(0.9)

            if index is None:
                model = get_active_model()
                if model:
                    index = model.current_step

            if isinstance(index, int):
                index = "{:05d}".format(index)

            if raster_shape is not None:
                img = plot_gauss_rbf(graph=graph, raster_shape=raster_shape)
                tf.io.write_file(f'{self.path_prefix}_raster_{index}.png', tf.image.encode_png(img))

            if plot_graph:
                plot_png(f'{self.path_prefix}_plot_{index}.png', graph=graph)

