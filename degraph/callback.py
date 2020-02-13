from .model import Callback, get_active_model

from datetime import datetime
from typing import Optional, Callable, Sequence, Union


class SnapshotCallback(Callback):
    """
    A callback that invokes a given callable at regular time intervals when a step is completed (i.e. equivalent with
    the callback's method on_training_step_end.
    """

    def __init__(self, fun: Callable, *, interval: Optional[float] = None):
        """
        Init callback.
        :param fun: The callable to be invoked.
        :param interval: Optional float value representing the interval in seconds between the invocations. If None
        the callable is invoked at each step.
        """
        super().__init__()
        self.fun = fun
        self.last_op = None
        self.interval = interval

    def on_training_step_end(self):
        interval = self.interval
        if interval is not None:
            now = datetime.now()
            if self.last_op is not None:
                if (now - self.last_op).total_seconds() < interval:
                    return
            self.last_op = now
        self.fun()


class GraphSnapshot(SnapshotCallback):
    def __init__(self, graph, *, interval: Optional[float] = None):
        """
        Init callback.
        :param graph: The graph to be monitored.
        :param interval: Optional float value representing the interval in seconds between the snapshots. If None
        the graph is saved at each step.
        """
        super().__init__(fun=self._run, interval=interval)
        self.graph = graph
        self.history = []

    def init(self, model):
        self.reset()
        super().init(model)

    def reset(self):
        self.history = []

    def _run(self):
        graph = self.graph
        if graph is None:
            return
        graph = graph.copy(static=True)
        model = get_active_model()
        self.history.append({'step': model.current_step, 'graph': graph})


class AlternateTrainingCallback(Callback):
    """
    A callback that sets the trainable state alternatively and mutually exclusively between two sets of entities.
    This component mimics the so called alternating gradient descent.
    """

    @staticmethod
    def _get_group(value):
        if isinstance(value, str):
            return [value]
        return list(value)

    def __init__(self, group1: Union[str, Sequence[str]], group2: Union[str, Sequence[str]], period: int = 1):
        """
        Init callback.
        :param group1: The string name (or a sequence of names) of the entity belonging to the first group.
        :param group2: Identifier for the entities of the second group.
        :param period: The period in terms of training steps for the switching.
        """

        if period <= 0:
            raise ValueError(f'Invalid period: {period}')

        super().__init__()
        self.period = period
        self.counter = 0
        self.status = False
        self.groups = self._get_group(group1), self._get_group(group2)

    def on_training_step_end(self):
        self.counter += 1
        if self.counter % self.period != 0:
            return
        self.counter = 0
        self.status = not self.status
        status = self.status

        model = get_active_model()
        for k in self.groups[0]:
            model.entity(k).trainable = status
        for k in self.groups[1]:
            model.entity(k).trainable = not status

