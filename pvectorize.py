from functools import reduce
from itertools import product
from multiprocessing import Pool
from numpy import fromiter


def pvectorize(f, otypes=None, doc=None, pool=None):
    def vectorized(*args):
        n = len(args)
        if n == 0:
            return f()
        shape = args[0].shape
        for a in args:
            if a.shape != shape:
                raise ValueError(
                    'operands could not be broadcast together with shapes ' +
                    ' '.join('(' + ','.join(str(x) for x in a.shape) + ')' for a in args))
        _otypes = otypes or [a.dtype for a in args]
        _pool = pool or Pool() 
        output = _pool.starmap(f, (tuple(reduce(lambda acc, x: acc[x], t, a) for a in args)
                                   for t in product(*(list(range(x)) for x in shape))))
        if pool is None:
            _pool.close()
        if n == 1:
            return fromiter(output, dtype=_otypes[0]).reshape(shape)
        return tuple(fromiter(it, dtype=_otypes[i]).reshape(shape)
                     for i, it in enumerate(zip(*output)))
    vectorized.__doc__ = doc if doc is not None else f.__doc__
    return vectorized
