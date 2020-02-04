from .math import reduce_any_nan
from .utils import export
from .types import TensorRef

import tensorflow as tf
import pandas as pd
from tqdm import tqdm

from typing import Tuple, Sequence, List, Optional, Union, Callable, Dict, Any
import abc
from datetime import datetime
from itertools import chain
import itertools
from contextlib import contextmanager, nullcontext
from collections import OrderedDict


class ModelException(Exception):
    pass


def expand_tensor_ref(ref):
    if ref is None:
        return None
    if isinstance(ref, Callable):
        return ref()
    if isinstance(ref, (tf.Tensor, tf.Variable)):
        return ref
    raise ValueError(f'Invalid tensor ref type: {type(ref)}')


_entity_name_counter = itertools.count()


def create_entity_name(value: Optional[str], prefix: str) -> str:
    if value is None:
        assert isinstance(prefix, str)
        idx = next(_entity_name_counter)
        return f'{prefix}{idx}' if prefix.endswith('_') else f'{prefix}_{idx}'

    if not value:
        raise ValueError('Invalid entity name')
    return value


class Entity(abc.ABC):  # TODO extend tf.Module?
    """
    Base class for graph/model entities.
    """

    def __init__(self, name=None):
        """
        Init the entity. The constructor is responsible for the creation of the entity's name.
        :param name:
        """
        self._name = create_entity_name(name, 'entity')

    # TODO add_dep (add dependency ref)

    def add_to_active_model(self):
        model = get_active_model()
        if model:
            model.add(self)

    @property
    def name(self) -> str:
        """
        Entity's name. It's a unique identifier for the current entity with respect to the model containing such object.
        :return:
        """
        return self._name

    def run(self):
        """
        Run entity contribution, this method must be invoked under an active model (see Model.as_active).
        :return:
        """
        pass

    def get_trainable_variables(self) -> Optional[Sequence[tf.Tensor]]:
        """
        Get a list of trainable variables relative to this entity. Return None is no trainable variables are present.
        :return:
        """
        return None


class SingleVarContrib(Entity):
    """
    An entity contributing to a single variable of the execution context.
    """

    @abc.abstractmethod
    def get_value(self) -> Union[float, tf.Tensor]:
        """
        Compute variable value.
        :return: The value of the variable.
        """
        return 0.

    def get_value_ref(self) -> TensorRef:
        def fun():
            model = get_active_model()
            assert model
            return model.runtime_context[self.name]
        return fun

    def run(self):
        get_active_model().runtime_context[self.name] = self.get_value()


@export
class Callback:
    def __init__(self):
        self.model = None

    def init(self, model):
        self.model = model

    def on_training_step_begin(self):
        pass

    def on_training_step_end(self):
        pass


_current_training_model = None


@export
class Model:
    LOSS_VAR_NAME = 'loss'

    def __init__(self):
        self._entities_map = OrderedDict()
        self.summary_writer_factory: Optional[tf.summary.SummaryWriter] = None
        self.current_step = None    # TODO create tensorref?

        self.runtime_context = {}
        self._current_mode = None
        self._current_history_rec = None

        # Variable storing the active gradient tape, to be used by stop_gradient_tape()
        self._current_gradient_tape = None

    @property
    def mode(self):
        """
        Current running mode, in training mode this is set to 'training'.
        :return:
        """
        return self._current_mode

    def reset(self):
        """
        Reset the runtime context and the history record.
        :return:
        """
        self.runtime_context = {}
        self._current_history_rec = {}

    def add(self, entity: Entity):
        """
        Add the given entity to the current model.
        :param entity:
        :return:
        """
        if entity.name in self._entities_map:
            raise ValueError(f'Duplicated entity name: {entity.name}')
        self._entities_map[entity.name] = entity

    @property
    def entities(self):
        return self._entities_map.values()

    def entity(self, name) -> Entity:
        """
        Get entity by name.
        :param name:
        :return:
        """
        return self._entities_map[name]

    @contextmanager
    def as_active(self):
        """
        Create a context manager that marks this model as the active one. The active model object can be obtained by
        invoking get_active_model().
        :return: A context manager.
        """
        global _current_training_model
        prev = _current_training_model
        _current_training_model = self
        try:
            yield
        finally:
            _current_training_model = prev

    def run(self, to_expand: Optional[TensorRef] = None) -> Optional[tf.Tensor]:
        """
        Execute the entities belonging to this model, at the end of the operations, the tensor ref given in the
        parameter to_expand is expanded and returned.
        :param to_expand:
        :return:
        """
        # TODO Param to_expand, allow dict/list of TensorRefs
        with self.as_active():
            self.runtime_context = {}
            for obj in self.entities:
                obj.run()
            if to_expand is not None:
                return expand_tensor_ref(to_expand)

    def get_trainable_vars(self) -> List[tf.Variable]:
        """
        Get a list of all trainable variable from the entities belonging to this model.
        :return: A list of instances of tf.Variable.
        """
        ents = filter(lambda ee: isinstance(ee, Entity), self.entities)
        trainable_vars = filter(lambda val: val is not None, [ent.get_trainable_variables() for ent in ents])
        trainable_vars = list(chain(*trainable_vars))
        var_names = list(map(lambda var: var.name, trainable_vars))
        if len(var_names) != len(set(var_names)):
            raise ModelException()
        return trainable_vars

    def update_history_rec(self, updates: Dict[str, Any]):
        history = self._current_history_rec
        if history is not None:
            history.update(updates)

    def compute_grad(self, loss: TensorRef = None) -> List[Tuple[tf.Tensor, tf.Variable]]:
        """
        Compute the layer's gradient and get a list of tuples (tensor, variable) compatible with the optimizer
        call apply_gradients.
        :return: A list of tuples (tensor, variable).
        """
        if loss is None:
            loss = self.runtime_context.get(self.LOSS_VAR_NAME)
        if loss is None:
            raise ValueError('No loss variable provided')

        trainable_vars = self.get_trainable_vars()
        with tf.GradientTape(watch_accessed_variables=False) as g:
            self._current_gradient_tape = g
            try:
                g.watch(trainable_vars)
                self.run()
                loss = expand_tensor_ref(loss)
            finally:
                self._current_gradient_tape = None
        assert not reduce_any_nan(loss)

        self.update_history_rec({'loss': float(loss)})

        grads = g.gradient(target=loss, sources=trainable_vars)
        return list(zip(grads, trainable_vars))

    def _get_summary_writer(self) -> tf.summary.SummaryWriter:
        factory = self.summary_writer_factory
        if factory is not None:
            return factory()
        return tf.summary.create_noop_writer()

    def _get_step_display_info(self):
        return OrderedDict((key, self._current_history_rec.get(key)) for key in ('step', 'loss'))

    def _fit(self, steps: int, loss: Optional[TensorRef], optimizer, callbacks: List[Callback]) -> pd.DataFrame:
        for c in callbacks:
            c.init(model=self)  # TODO reset (representing the start of a trial)

        history = []
        summary_writer = self._get_summary_writer()  # TODO Optionally get from function params
        with summary_writer.as_default():
            iterator = tqdm(range(steps), desc='fit')
            for k in iterator:
                self.reset()
                self._current_history_rec.update({'step': k, 'datetime': datetime.now()})

                self.current_step = k
                tf.summary.experimental.set_step(k)
                for c in callbacks:
                    c.on_training_step_begin()

                grads = self.compute_grad(loss=loss)
                invalid_grads = list(filter(lambda item: reduce_any_nan(item[0]), grads))
                assert len(invalid_grads) == 0
                iterator.set_postfix(ordered_dict=self._get_step_display_info())

                with tf.name_scope('grads'):
                    for pair in grads:
                        tf.summary.histogram(name=pair[1].name, data=pair[0], step=k)

                optimizer.apply_gradients(grads_and_vars=grads)

                for c in callbacks:
                    c.on_training_step_end()

                history.append(self._current_history_rec.copy())

        summary_writer.flush()
        history = pd.DataFrame.from_records(history)
        history.set_index('step', drop=False)
        return history

    def fit(self, steps, *, loss: Optional[TensorRef] = None, optimizer,
            callbacks: Union[List[Callback], Callback, None] = None) -> pd.DataFrame:
        """
        Run the "training" (fitting) algorithm. During the process the active model
        (the one returned by get_active_model()) is set to the current Model instance.
        :param steps: The number of iterations.
        :param loss: A tensor ref of the loss function.
        :param optimizer: The optimizer object, it must support the call apply_gradients.
        :param callbacks: A callback or a list of callbacks.
        :return: A Pandas DataFrame containing the history of the optimization process.
        """
        if callbacks is None:
            callbacks = []
        elif isinstance(callbacks, Callback):
            callbacks = [callbacks]
        if not isinstance(callbacks, list):
            raise ValueError(f'Unsupported callback type: {type(callbacks)}')

        if optimizer is None:
            raise ValueError('Invalid optimizer')
        if not hasattr(optimizer, 'apply_gradients'):
            raise ValueError('Optimizer not implementing apply_gradients method')
        # optimizer_orig = optimizer
        # optimizer = optimizer_orig.from_config(optimizer_orig.get_config())
        # TODO Optionally pass a validation_loss (slow) to be evaluated at the end of each trial? (method fit_trials)

        prev_mode = self._current_mode
        try:
            self._current_mode = 'training'
            with self.as_active():
                ret = self._fit(steps=steps, loss=loss, optimizer=optimizer, callbacks=callbacks)
        finally:
            self._current_mode = prev_mode
            self.reset()
            tf.summary.experimental.set_step(None)

        return ret


@export
def get_active_model() -> Optional[Model]:
    """
    Get the current active model. Model are marked as active using the context manager Model.as_active().
    :return: The object of the current model or None is no Model is active.
    """
    return _current_training_model


@export
def stop_gradient_tape(cond: bool = True):
    """
    Create a context manager that prevent the gradient tape from recording (conditionally to the parameter cond)
    when in training context. When not in training context a nullcontext is returned.
    :param cond: A boolean value that when true enables the stopping action.
    :return: A context manager.
    """

    if not cond:
        return nullcontext()

    model = get_active_model()
    if model is None:
        return nullcontext()
    if model._current_gradient_tape is None:
        return nullcontext()
    return model._current_gradient_tape.stop_recording()

