import degraph as dg
from degraph.plot import plot_matplt
import tensorflow as tf
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import os
import traceback
from datetime import datetime
from typing import Optional

matplotlib.use("Agg")


def create_summary_writer_factory(output_path: str):
    """
    Create summary writer for Tensorboard.
    :return:
    """
    if output_path is None:
        return None

    path = os.path.join(output_path, f"logs")
    print(f"logs path: {path}")

    def fun():
        return tf.summary.create_file_writer(path)
    return fun


def create_sample_graph():
    """
    Create a sample graph.
    :return: Return an adjacency matrix.
    """
    g = nx.balanced_tree(3, 4)
    # g = nx.grid_2d_graph(5, 5)
    # g = nx.karate_club_graph()
    m = nx.to_numpy_matrix(g)
    return m


def create_path_length_loss(raster_shape):
    """
    Create loss component that penalizes the length of the edges.
    :return:
    """

    @tf.function
    def path_length_loss(edges: tf.Tensor):
        loss_factor = 1.
        edges *= tf.convert_to_tensor(raster_shape, dtype=dg.TF_FLOAT)
        lengths = dg.math.path_length(edges)
        # lengths = tf.square(lengths)
        # lengths = tf.abs(lengths - 5.)
        lengths = tf.reduce_mean(lengths) * loss_factor
        return lengths
    return path_length_loss


def create_matplt_plotter(graph: dg.GraphRepr, path_prefix: str):
    def fun(index=None):
        if index is None:
            model = dg.get_active_model()
            if model:
                index = model.current_step

        if isinstance(index, int):
            index = "{:05d}".format(index)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plot_matplt(graph, ax=ax)
        ax.autoscale()
        fig.savefig(f'{path_prefix}_plot_{index}.png')
    return fun


def run_graph_fit(adjacency, dim: int = 3, output_path: Optional[str] = None):
    """
    Run a fitting experiment using a graph which connectivity is represented by the given adjacency matrix.
    :param adjacency: The adjacency matrix of the input graph.
    :param dim: The number of dimensions of the space, either 2 or 3.
    :param output_path: Optionally the path for Tensorboard logs.
    """
    raster_shape = (128,) * dim  # Shape of the raster (image when dim=2) where the RBFs are plotted

    graph = dg.GraphRepr(adjacency=adjacency, dim=len(raster_shape))    # Create graph

    model = dg.Model()
    with model.as_active():
        # Create entity representing edges
        edges = dg.piecewise_linear_edges(graph, steps=25, space_radius=10., name='ed', trainable=True)
        # Create entity representing vertexes
        vx = dg.vertexes(graph, trainable=True, name='vx')

        # Create a raster where Wendland RBF is used to represent the vertexes interaction.
        raster = dg.aggregate_rasters([
            dg.rbf_net_raster(vx, shape=raster_shape, rbf='wendland', spread=3.0, peak=1.0)
        ], bias=-1.)

        dg.summary_histogram(raster, scope='raster_histo')

        # Loss contributions
        losses = {
            'metrics.raster_loss': dg.sse_loss(raster),
            'metrics.bounds_loss': dg.unit_sphere_bounds_loss(edges),
            'metrics.path_length_loss': dg.lambda_(create_path_length_loss(raster_shape), edges)
        }
        for loss_ in losses.items():
            dg.summary_scalar(loss_[1], scope=loss_[0])     # Loss contribution summary (Tensorboard)
        loss = dg.reduce_sum(losses.values(), name='loss')  # Aggregate the individual losses
        dg.summary_scalar(loss, scope='metrics.loss')       # Summarise the overall loss on Tensorboard

    session_path = None
    if output_path is not None:
        fit_session = datetime.now().strftime('%Y%m%d%H%m%S')
        print(f'Current fit session: {fit_session}')
        session_path = os.path.join(output_path, fit_session)
        os.mkdir(session_path)

    # plotter = create_matplt_plotter(graph, path_prefix=os.path.join(session_path, 'step'))
    model.summary_writer_factory = create_summary_writer_factory(session_path)

    graph_snapshots = dg.callback.GraphSnapshot(graph, interval=5)
    # callbacks = [
    #    dg.callback.SnapshotCallback(plotter, interval=5),  # Snapshot callback, plot the status every X seconds
    # ]

    try:
        # plotter(index='init')  # Plot init state

        # Create optimizer (SGD)
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-1, decay=2., momentum=0.9, clipnorm=1.0)
        # Run the fitting
        history = model.fit(1000, loss=loss, optimizer=optimizer, callbacks=graph_snapshots)
        # Now in graph_snapshots.history we have the snapshots of the graph at the various time steps
    except AssertionError:
        traceback.print_exc()

    # plotter(index='end')    # Plot the end state
    print(graph.positions)


run_graph_fit(adjacency=create_sample_graph())

