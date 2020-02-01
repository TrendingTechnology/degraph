from typing import Tuple, Union, Sequence, Callable, List

import pandas as pd
import numpy as np
import tensorflow as tf


Adjacency = Union[pd.DataFrame, np.ndarray]
Aggregate = Callable[[List[tf.Tensor]], tf.Tensor]
StaticShape = Tuple[int, ...]
RankStaticShape = Tuple[Union[int, tf.Tensor], ...]
OptLayerNames = Union[str, Sequence[str], None]
RBF = Callable[[tf.Tensor], tf.Tensor]


"""
TensorRef is a type for a lazy tensor. Lazy because its value depends on the current execution context. The execution
context is a dictionary mapping tensor name to its value. The Model object contains the reference to the dictionary
representing the mapping. Instances of TensorRef are resolved during the execution of a model (eg. during the
optimization).
"""
TensorRef = Union[tf.Tensor, Callable[[], tf.Tensor]]

