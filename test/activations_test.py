from deep_learning.activations import relu


def test_relu():
    assert relu(-5) == 0
    assert relu(0) == 0
    assert relu(10) == 10
