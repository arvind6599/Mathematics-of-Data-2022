import numpy as np
from typing import Callable, Optional, List, Union
from dataclasses import dataclass, astuple


class Matrix(np.ndarray):
    pass


@dataclass
class OptState:
    X: Matrix
    AX: Matrix
    k: int

    def __iter__(self):
        return iter(astuple(self))


@dataclass
class Function:
    f: Callable[[Matrix], float]
    grad: Optional[Callable[[Matrix], Matrix]] = None
    kappa: int = None

    def __call__(self, AX):
        return self.f(AX)


@dataclass
class OptAlgorithm:
    name: str
    init_state: Callable[[Function], OptState] = None
    state_update: Callable[[Function, OptState], OptState] = None
