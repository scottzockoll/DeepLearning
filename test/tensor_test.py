import deep_learning as dl
from deep_learning.utils import zeros, ones, fill
from .testing_tools import get_positive_tuple

t1 = dl.Tensor([1, 2, 3])
t2 = dl.Tensor([1, 1, 1])
m1 = dl.Tensor([
    [1, 2, 3],
    [4, 5, 6]
])
m1_T = dl.Tensor([
    [1, 4],
    [2, 5],
    [3, 6]
])


def test_eq():
    assert dl.Tensor([]) == dl.Tensor([])
    assert dl.Tensor([]) != dl.Tensor([0])
    assert t1 == dl.Tensor([1, 2, 3])
    assert t1 != dl.Tensor([0, 2, 3])
    shape = get_positive_tuple()
    assert zeros(shape) == zeros(shape)
    assert ones(shape) != zeros(shape)


def test_sub():
    assert t1 - t2 == dl.Tensor([0, 1, 2])


def test_T():
    assert m1.T == m1_T


def test_len():
    assert len(dl.Tensor([])) == 0
    assert len(t1) == 3


def test_apply():
    assert t2.apply(lambda x: x + 1) == fill(2, t2.shape)
