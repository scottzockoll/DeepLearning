import deep_learning as dl
from deep_learning.activations import relu

from deep_learning.utils import ones
from deep_learning.validation import assert_supported_params


# Assumes single chain graph
class ComputationalGraph:
    def __init__(self, input_gate):
        self.input_gate = input_gate
        self.graph = {}
        self.ordered_gates = []
        self.add_vertex(input_gate)

    def add_vertex(self, u):
        self.graph[u] = set()
        self.ordered_gates.append(u)

    def add_edge(self, u, v):
        self.graph.setdefault(u, set()).add(v)
        self.add_vertex(v)

    def __len__(self):
        return len(self.graph)

    def forward(self, input_value):
        h = input_value
        for k in self.ordered_gates:
            h = k.forward(h)
        return h

    def backward(self):
        grad = 1
        for k in reversed(self.ordered_gates):
            grad = k.backward(grad)
        return grad


class ConstantGate:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        assert isinstance(self.x, dl.Tensor)
        return x

    def backward(self, dz):
        return self.x


class AffineGate:
    def __init__(self, input_size, output_size, w_init_strat):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = self.init_weights(w_init_strat)
        self.x = None

    def init_weights(self, w_init_strat: str) -> dl.Tensor:
        shape_of_weights = (self.input_size, self.output_size) if self.output_size != 1 else (self.input_size,)
        supported_strats = {
            "ones": ones
        }
        assert_supported_params(
            supported_strats, w_init_strat, "Weight initialization strategy"
        )
        generate_weights = supported_strats[w_init_strat]
        return dl.Tensor(generate_weights(shape_of_weights))

    def forward(self, x):
        self.x = x
        if len(x) != self.input_size:
            raise ValueError(
                "Input should have shape {} but has shape {}".format(
                    self.input_size, len(x)
                )
            )
        result = self.weights.T @ x
        return result

    def backward(self, dz):
        print('dz: {}'.format(dz))
        grad_weights = self.x * dz
        print(grad_weights)
        self.weights = self.weights - (dl.Tensor([.1]) * ((dl.Tensor([-1])) * grad_weights))
        print('weights updated to: {}'.format(self.weights))
        return self.weights.T * dz


class ReluGate:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        print('relu: {}'.format(x.apply(relu)))
        return x.apply(relu)

    def backward(self, dz):
        def f(x):
            return 0 if x <= 0 else 1

        print('relu backward: {}'.format(self.x.apply(f)))
        return self.x.apply(f) * dz


class DummyLoss:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        # no need to return because
        # this is always at the end of the network

    def backward(self, dz):
        loss = (dl.Tensor([50]) - self.x).apply(abs)
        print('loss of: {}'.format(loss))
        return dl.Tensor([1])


affine1 = AffineGate(5, 1, w_init_strat='ones')
relu1 = ReluGate()
loss = DummyLoss()

graph = ComputationalGraph(affine1)
graph.add_edge(affine1, relu1)
graph.add_edge(relu1, loss)

inputs = dl.Tensor([2, -3, 4, 5, 1])

for _ in range(10):
    graph.forward(inputs)
    graph.backward()
