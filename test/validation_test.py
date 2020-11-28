import pytest

from test.testing_tools import get_positive_tuple
from utils import zeros
from validation import assert_n_dims

t1 = zeros(get_positive_tuple())


def test_assert_n_dims():
    with pytest.raises(ValueError):
        assert_n_dims(t1, t1.n_dims + 1)
    assert_n_dims(t1, t1.n_dims)
