import itertools

import deep_learning as dl
from deep_learning.utils import fill, zeros
from deep_learning.validation import (assert_n_dims, assert_same_shape,
                                      is_scalar)


class Tensor:
    def __init__(self, m):
        self.m = m
        self.shape = self.get_tensor_shape(m)
        self.n_dims = len(self.shape)
        self.iter_count = -1

    @property
    def T(self):
        result = zeros(self.shape[::-1])
        it = self.index_iterator()
        for tup in it():
            value = self.m
            for i in tup:
                value = value[i]
            result[tup[::-1]] = value

        return result

    def __str__(self):
        if len(self.shape) == 1:
            return str(self.m)
        result = ""
        for row in self.m:
            result += str(row) + "\n"
        return result

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_count == self.__len__() - 1:
            self.iter_count = -1
            raise StopIteration
        else:
            self.iter_count += 1
            return self.__getitem__(self.iter_count)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            result = self.m
            for i in index:
                result = result[i]
            return result
        else:
            return self.m[index]

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            target = self.m
            for i in range(len(index) - 1):
                target = target[index[i]]
            target[index[-1]] = value

        else:
            self.m[index] = value

    def __len__(self):
        return len(self.m)

    def __matmul__(self, other):
        if self.n_dims == 1 and other.n_dims == 1:
            if len(self.shape) != len(other.shape):
                raise ValueError(
                    "Tried to multiply two 1d vectors of shapes {} and {}".format(
                        len(self.shape), len(other)
                    )
                )
            return sum([i * j for i, j in zip(self.m, other)])

        if self.shape[1] != other.shape[0]:
            raise ValueError(
                "Tried to multiple two matricies with sizes: {} and {}".format(
                    self.shape, other.shape
                )
            )
        left = dl.Tensor([self.m]) if len(self.shape) == 1 else dl.Tensor(self.m)
        other = dl.Tensor([[i] for i in other]) if len(other.shape) == 1 else other

        result = zeros((self.shape[0], other.shape[1]))
        it = result.index_iterator()
        for i, j in it():
            result[i][j] = sum([left[i][k] * other[k][j] for k in range(self.shape[1])])
        if result.n_dims == 2 and result.shape[1] == 1:
            flat_result = []
            for e in result:
                flat_result.extend(e)
            return dl.Tensor(flat_result)
        else:
            return result

    def __add__(self, other):
        assert_same_shape(self, other)
        result = zeros(self.shape)
        it = self.index_iterator()
        for tup in it():
            result[tup] = self[tup] + other[tup]
        return result

    def __sub__(self, other):
        assert_same_shape(self, other)
        result = zeros(self.shape)
        it = self.index_iterator()
        for tup in it():
            result[tup] = self[tup] - other[tup]
        return result

    def __mul__(self, other):
        left_term = self
        right_term = other
        if is_scalar(self):
            left_term = fill(self[0], other.shape)
        if is_scalar(other):
            right_term = fill(other[0], left_term.shape)
        assert_same_shape(left_term, right_term)
        result = zeros(left_term.shape)
        it = left_term.index_iterator()
        for tup in it():
            result[tup] = left_term[tup] * right_term[tup]
        return result

    def __eq__(self, other):
        if self.shape != other.shape:
            return False
        it = self.index_iterator()
        for ind in it():
            if self[ind] != other[ind]:
                return False
        return True

    def index_iterator(self):
        shape = self.shape

        def it():
            ranges = [range(i) for i in shape]
            for i in itertools.product(*ranges):
                yield i

        return it

    @staticmethod
    def get_tensor_shape(t):
        result = []
        while isinstance(t, list) or isinstance(t, dl.Tensor):
            if isinstance(t, dl.Tensor):
                result.extend(t.shape)
                break
            result.append(len(t))
            if t:
                t = t[0]
            else:
                t = False
        return tuple(result)

    # x <dl.Tensor> of shape (n,)
    def v_append(self, x):
        assert_n_dims(x, 1)
        if self.shape[0] != x.shape[0]:
            raise ValueError(
                "The shape of x must match the first dimension of the dl.Tensor but x has {} shape "
                "and dl.Tensor has {} shape".format(x.shape, self.shape)
            )
        if self.n_dims == 1:
            result = [x_i for x_i in self.m]
            result.append(x[0])
            return dl.Tensor(result)
        else:
            result = []
            for row, x_i in zip(self.m, x):
                new_row = []
                new_row.extend(row)
                new_row.append(x_i)
                result.append(dl.Tensor(new_row))
            return dl.Tensor(result)

    # could -> dl.Tensor for some reason
    # also couldn't do f: function
    def apply(self, f):
        it = self.index_iterator()
        result = zeros(self.shape)
        for ind in it():
            result[ind] = f(self[ind])

        return result


# value <any>
# shape <tuple>
# Returns a dl.Tensor full of value of a certain shape
