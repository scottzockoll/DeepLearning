import deep_learning as dl


class ComputationalGraph:
    def __init__(self, input_gates):
        self.input_gates = (
            input_gates if isinstance(input_gates, list) else list(input_gates)
        )
        self.graph = {}
        for gate in self.input_gates:
            self.add_vertex(gate)

    def add_vertex(self, u):
        self.graph[u] = set()

    def add_edge(self, u, v):
        self.graph.setdefault(u, set()).add(v)
        self.add_vertex(v)

    def __len__(self):
        return len(self.graph)

    def topological_sort(self):
        seen = set()
        stack = []  # path variable is gone, stack and order are new
        order = []  # order will be in reverse order at first
        q = list(self.input_gates)
        while q:
            v = q.pop()
            if v not in seen:
                seen.add(v)  # no need to append to path any more
                q.extend(self.graph[v])

                while stack and v not in self.graph[stack[-1]]:  # new stuff here!
                    order.append(stack.pop())
                stack.append(v)

        return stack + order[::-1]  # new return value!

    # TODO this does not preserve the order
    def get_parents(self, gate):
        result = []
        for u, vertices in self.graph.items():
            if gate in vertices:
                result.append(u)

        return result

    def get_call_stack(self, gate):
        parents = self.get_parents(gate)
        if len(parents) != 0:
            other_parents = []
            for parent in parents:
                other_parents.extend(self.get_call_stack(parent))
            return other_parents + [parent for parent in parents]
        else:
            return []

    def forward(self, input_values):
        results = []
        for gate in self.topological_sort():
            print(gate)
            if gate in self.input_gates:
                results.append(gate.forward(input_values.pop(0)))
                # print(results)
            else:
                inputs = [results.pop(0) for _ in range(gate.n_inputs)]
                results.extend(gate.forward(*inputs))


class Gate:
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.parents = []

    def set_parent(self, parent):
        self.parents.append(parent)


class MultiplyGate(Gate):
    def __init__(self):
        super().__init__(2)
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x @ y

    def backward(self, dz):
        dx = dz * self.y
        dy = dz * self.x
        return dx, dy


class AddGate(Gate):
    def __init__(self):
        super().__init__(2)
        self.x = None
        self.y = None

    def forward(self, x, y):
        return x + y

    def backward(self, dz):
        return dz, dz


class ConstantGate(Gate):
    def __init__(self):
        super().__init__(1)
        self.x = None

    def forward(self, x):
        self.x = x
        assert isinstance(self.x, dl.Tensor)
        return x

    def backward(self, dz):
        return self.x


# t1 = dl.Tensor([1, 2, 3])
# t2 = dl.Tensor([1, 1, 1])
# t1 = dl.Tensor([-2, 2])
# t2 = dl.Tensor([5, 2])
# t3 = dl.Tensor([-4, 2])

add = AddGate()
mul = MultiplyGate()
x = ConstantGate()
y = ConstantGate()
z = ConstantGate()

graph = ComputationalGraph({x, y, z})
graph.add_edge(x, add)
graph.add_edge(y, add)
graph.add_edge(add, mul)
graph.add_edge(z, mul)
# graph.forward([dl.Tensor([-2]), dl.Tensor([5]), dl.Tensor([-4])])

# mul.forward(add.forward(t1, t2), t3)
# print(mul.backward(dl.Tensor([1])))
# print(add.backward(mul.backward(dl.Tensor([1])))[0])

# graph = ComputationalGraph()
# graph.add_edge(add, mul)
# print(graph.topological_sort())


# values = {
#     'a': ['c'],
#     'b': ['c', 'd'],
#     'c': ['d'],
#     'd': ['e', 'f'],
#     'e': [],
#     'f': ['g'],
#     'g': []
# }
# graph = ComputationalGraph(['a', 'b'])
# for vertex, verticies in values.items():
#     graph.add_vertex(vertex)
#     for v in verticies:
#         graph.add_edge(vertex, v)
# print(graph.graph)
# print(graph.topological_sort())
