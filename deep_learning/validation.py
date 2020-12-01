def assert_n_dims(x, n_dims: int) -> None:
    if x.n_dims != n_dims:
        raise ValueError(
            "Tensor must have n_dims={} but has n_dims={}".format(n_dims, x.n_dims)
        )


# Asserts that param_str in supported
# class_name describes what supported
# the keys in supported should be lowercase
def assert_supported_params(supported: dict, param_str: str, class_name: str) -> None:
    if param_str.lower() not in supported:
        raise ValueError("{} is not a supported {}".format(param_str, class_name))


def assert_same_shape(a, b):
    if a.shape != b.shape:
        raise ValueError("Tensor does not have the same shape. Left has {} and right has {}".format(a.shape, b.shape))


# Returns true if Tensor is a scalar
def is_scalar(x):
    return True if x.shape == (1,) else False
