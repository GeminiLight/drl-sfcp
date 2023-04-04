import itertools
from .solver import Solver
from .heuristic import *
from .learning import *
from .registry import REGISTRY, register, get

from . import heuristic, learning


SOLVERS = {
    'heuristic': tuple(heuristic.__all__),
    'learning': tuple(learning.__all__),
}
SOLVERS['all'] = tuple(itertools.chain.from_iterable(SOLVERS.values()))

__all__ = list(SOLVERS['all']) + [
    'Solver',
    'REGISTRY',
    'register',
    'get',
]
