import deep_learning as dl
from deep_learning.utils import ones
from deep_learning.activations import relu
from deep_learning.validation import assert_supported_params


class Layer:
    def __init__(self, n_nodes, act_func, input_size, has_bias=True, w_init_strat='ones'):
        self.act_func = self.validate_act_func(act_func)
        self.n_nodes = n_nodes
        self.input_size = input_size
        self.has_bias = has_bias
        self.weights = self.init_weights(w_init_strat, has_bias)

    def forward(self, x: dl.Tensor) -> dl.Tensor:
        if len(x) != self.input_size:
            raise ValueError('Input should have shape {} but has shape {}'.format(self.input_size, len(x)))
        x = self.add_bias(x)
        return (self.weights.T * x).apply(self.act_func)

    def add_bias(self, x: dl.Tensor) -> dl.Tensor:
        if self.has_bias:
            return x.v_append(dl.Tensor([1 for _ in range(len(x))]))
        else:
            return x

    def validate_act_func(self, act_func: str):
        supported_funcs = {
            'relu': relu
        }
        assert_supported_params(supported_funcs, act_func, 'Activation function')
        return supported_funcs[act_func]

    def init_weights(self, w_init_strat: str, has_bias: bool) -> dl.Tensor:
        shape_of_weights = (self.input_size + 1 if has_bias else self.input_size, self.n_nodes)
        # w_init_strat needs to be a function that returns a matrix of shape: shape_of_weights
        supported_strats = {
            'ones': ones
        }
        assert_supported_params(supported_strats, w_init_strat, 'Weight initialization strategy')
        generate_weights = supported_strats[w_init_strat]
        return dl.Tensor(generate_weights(shape_of_weights))


lay1 = Layer(3, 'relu', 5, w_init_strat='ones')
lay2 = Layer(2, 'relu', 3)
result = lay2.forward(lay1.forward(dl.Tensor([1, 2, 3, 4, 5])))
print(lay2.weights.T)
print(result)
