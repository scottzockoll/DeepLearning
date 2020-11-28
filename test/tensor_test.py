from tensor import Tensor
from utils import zeros, ones
from .testing_tools import get_positive_tuple

t1 = Tensor([1, 2, 3])
t2 = Tensor([1, 1, 1])
m1 = Tensor([
    [1, 2, 3],
    [4, 5, 6]
])
m1_T = Tensor([
    [1, 4],
    [2, 5],
    [3, 6]
])


def test_eq():
    assert Tensor([]) == Tensor([])
    assert Tensor([]) != Tensor([0])
    assert t1 == Tensor([1, 2, 3])
    assert t1 != Tensor([0, 2, 3])
    shape = get_positive_tuple()
    assert zeros(shape) == zeros(shape)
    assert ones(shape) != zeros(shape)


def test_sub():
    assert t1 - t2 == Tensor([0, 1, 2])


def test_T():
    assert m1.T == m1_T

def test_oops():
    assert False
