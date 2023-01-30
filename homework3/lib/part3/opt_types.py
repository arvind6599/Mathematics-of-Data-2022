import numpy as np
from typing import Callable, Optional, List, Union
from dataclasses import dataclass, astuple


class Vector(np.ndarray):
    pass

class Matrix(np.ndarray):
    pass

    
@dataclass
class OptState:
    def __iter__(self):
        return iter(astuple(self))


@dataclass
class HCGM_state(OptState):
    x_k: Matrix
    k: int
    beta0: float


@dataclass
class Function:
    f: Callable[[HCGM_state], float]
    grad: Optional[Callable[[HCGM_state], OptState]] = None
    
    def __call__(self, x):
        return self.f(x)

    
@dataclass
class ConstrainedProblem:
    f: Function
    penalties: List[Function]
    C: float
    p: float
    opt_val: float
    X_true: Matrix

    def __iter__(self):
        return iter((self.f, self.penalties))
    
    def __call__(self, x):
        return self.f(x)


@dataclass
class OptAlgorithm:
    name: str
    init_state: Callable[[ConstrainedProblem, Vector], OptState] = None
    state_update: Callable[[ConstrainedProblem, OptState], OptState] = None
