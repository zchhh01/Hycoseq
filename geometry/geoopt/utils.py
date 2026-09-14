import itertools


def size2shape(*size):
    if len(size) == 1:
        value = size[0]
        return value if isinstance(value, tuple) else (value,)
    return tuple(size)


def broadcast_shapes(*shapes):
    result = []
    for dimensions in itertools.zip_longest(*map(reversed, shapes), fillvalue=1):
        dimension = 1
        for candidate in dimensions:
            if dimension != 1 and candidate != 1 and candidate != dimension:
                raise ValueError("shapes cannot be broadcast")
            dimension = max(dimension, candidate)
        result.append(dimension)
    return tuple(reversed(result))


def ismanifold(instance, cls):
    from .manifolds import Manifold

    if not issubclass(cls, Manifold):
        raise TypeError("cls must inherit Manifold")
    return isinstance(instance, cls)
