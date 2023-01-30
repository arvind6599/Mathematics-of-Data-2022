import numpy as np
from typing import Callable, Optional, List, Union
from dataclasses import dataclass, astuple


class Vector(np.ndarray):
    pass

class Matrix(np.ndarray):
    pass

@dataclass 
class DualVariable:
    y_1: Vector
    y_2: Vector
    y_3: Matrix

    def __add__(self, other):
        return DualVariable(self.y_1 + other.y_1, self.y_2 + other.y_2, self.y_3 + other.y_3)

    def __rmul__(self, other):
        return self.__lmul__(other)
    
    def __lmul__(self, other):
        if isinstance(other, list):
            return DualVariable(other[0]*self.y_1, other[1]*self.y_2, other[2]*self.y_3)
        else:
            return DualVariable(other*self.y_1, other*self.y_2, other*self.y_3)

    def __sub__(self, other):
        return self + -1*other

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
class ConstrainedProblem:
    f: Function
    penalties: List[Function]
    minimum: float = None

    def __iter__(self):
        return iter((self.f, self.penalties))
#    def __call__(self, x):
#        return self.f(x)
@dataclass
class OptState:
    x_k: Matrix

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

