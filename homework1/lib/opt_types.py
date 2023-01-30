import numpy as np
from typing import Callable, Optional, List, Union
from dataclasses import dataclass, astuple


class Vector(np.ndarray):
    pass

@dataclass
class Function:
    f: Callable[[Vector], float]
    grad: Optional[Callable[[Vector], Vector]] = None
    subgrad: Optional[Callable[[Vector], Vector]] = None
    i_grad: Optional[Callable[[int, Vector], Vector]] = None
    minimum: Optional[float] = None
    strng_cvx: Optional[float] = None
    lips_grad: Optional[float] = None
    n: Optional[int] = None
    L_max: Optional[float] = None
    prox: Optional[Callable[[float, Vector], Vector]] = None

    def __call__(self, x):
        return self.f(x)

@dataclass
class CompositeFunction:
    f: Function
    g: Function
    minimum: float = None

    def __iter__(self):
        return iter((self.f, self.g))
    
    def __call__(self, x):
        return self.f(x) + self.g(x)

@dataclass
class OptState:
    x_k: Vector

    def __iter__(self):
        return iter(astuple(self))

@dataclass
class OptAlgorithm:
    name: str
    init_state: Callable[[Union[Function, CompositeFunction], Vector], OptState] = None
    state_update: Callable[[Union[Function, CompositeFunction], OptState], OptState] = None

    
@dataclass
class RunTrace:
    sequence: List[Vector]
    values: List[Vector]


@dataclass
class Regularizer:
    g: Function
    lmda: float = None
    
    def __call__(self, x):
        return self.lmda*self.g(x)
    
    def prox(self, gamma, x):
        return self.g.prox(self.lmda * gamma, x)

