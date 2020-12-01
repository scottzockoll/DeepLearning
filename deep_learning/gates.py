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
        print('[Forward in AffineGate] Weights: {}, Computation: {}'.format(self.weights, result))
        return result

    def backward(self, dz):
        grad_weights = self.x * dz
        print('[Backward in AffineGate] Gradient of loss with respect to weights: {}'.format(grad_weights))
        self.weights = self.weights - (dl.Tensor([.1]) * grad_weights)
        result = self.weights.T * dz
        print('[Backward in AffineGate] Weights: {}, Gradient being passed back: {}'.format(self.weights, result))
        return result


class ReluGate:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        result = x.apply(relu)
        print('[Forward in ReluGate] Computation: {}'.format(result))
        return result

    def backward(self, dz):
        def f(x):
            return 0 if x <= 0 else 1

        result = self.x.apply(f) * dz
        print('[Backward in ReluGate] Gradient: {}'.format(result))
        return result


class DummyLoss:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        # no need to return because
        # this is always at the end of the network
        loss = abs((50 - self.x[0]))
        print('[Forward in DummyLoss] Computation: {}'.format(loss))
        return loss

    def backward(self, dz):
        # doesn't do anything with dz because DummyLoss is at the end of the network
        result = dl.Tensor([1]) if self.x[0] >= 50 else dl.Tensor([-1])
        print('[Backward in DummyLoss] Gradient: {}'.format(result))
        return result


affine1 = AffineGate(5, 1, w_init_strat='ones')
relu1 = ReluGate()
loss = DummyLoss()

graph = ComputationalGraph(affine1)
graph.add_edge(affine1, relu1)
graph.add_edge(relu1, loss)

inputs = dl.Tensor([2, -3, 4, 5, 1])

for _ in range(100):
    graph.forward(inputs)
    graph.backward()
