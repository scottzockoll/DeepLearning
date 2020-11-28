import random


def get_positive_tuple(low=1, high=10):
    return tuple(random.randint(low, high) for _ in range(random.randint(low, high)))
