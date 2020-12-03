import deep_learning as dl
from deep_learning.tensor import broadcast
from deep_learning.utils import fill

t1 = fill(1, shape=(2, 1))
t2 = fill(0, shape=(8, 4, 3))
# t1 = dl.Tensor([12,12,3])
# t2 = dl.Tensor([5,5,5,5])
print(t1.shape, t2.shape)
broadcast(t1, t2)
