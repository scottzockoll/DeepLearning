from test.testing_tools import get_positive_tuple

import pytest

from deep_learning.utils import zeros
from deep_learning.validation import assert_n_dims

t1 = zeros(get_positive_tuple())


def test_assert_n_dims():
    with pytest.raises(ValueError):
        assert_n_dims(t1, t1.n_dims + 1)
    assert_n_dims(t1, t1.n_dims)
