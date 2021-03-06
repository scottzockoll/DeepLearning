import copy

import deep_learning as dl

print(dir(dl))


# value <any>
# shape <tuple>
# Returns a Tensor full of value of a certain shape
def fill(value, shape):
    rev_shape = shape[::-1]
    result = [value for i in range(rev_shape[0])]
    for size in rev_shape[1:]:
        result = [copy.deepcopy(result) for i in range(size)]
    return dl.Tensor(result)


# shape <tuple>
# Return a Tensor full of zeros of a certain shape
def zeros(shape):
    return fill(0, shape)


def ones(shape):
    return fill(1, shape)
