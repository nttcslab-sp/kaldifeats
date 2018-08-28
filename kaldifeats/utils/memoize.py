from functools import WRAPPER_ASSIGNMENTS
from functools import partial

"""
Currently using this functools.lru_cache instead of this class


Source:

https://stackoverflow.com/questions/6394511/python-functools-wraps-equivalent-for-classes
"""


class Memoize:
    """Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    
    Example:
        >>> @Memoize
        >>> def f(a: int):
        ...     # do some heavy calculation...
        ...     return a
        >>> value = f()  # Performing function really
        >>> value = f()  # The cached value is derived
        
        Care not to use unhashable arguments for a wrapped function 
        like the following function,
        
        >>> import numpy
        >>> @Memoize
        >>> def f(array: numpy.ndarray):
        ...     # Do some heavy calculation
        ...     return arrray
        
        For example, list, dict, numpy.ndarray are unhashable type, 
        basically hash keys must be immutable, 
        so they cannot be cached in memory.
        I belive the best way to solve this is using joblib.cache, see below.
        
        https://pythonhosted.org/joblib/memory.html
    """
    def __init__(self, func, assigned=WRAPPER_ASSIGNMENTS):
        for attr in assigned:
            setattr(self, attr, getattr(func, attr))
        self.func = func
        self.cache = {}

    def __call__(self, *args, **kwargs):
        hashable_kwargs = tuple(sorted(kwargs.items(), key=lambda x: x[0]))

        try:
            return self.cache[(args, hashable_kwargs)]
        except KeyError:
            value = self.func(*args, **kwargs)
            self.cache[(args, hashable_kwargs)] = value
            return value
        except TypeError:
            # uncacheable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            return self.func(*args, **kwargs)

    def __repr__(self):
        return self.func.__repr__() +\
            ' (Wrapped by ' + self.__class__.__name__ + ')'

    def __get__(self, obj, objtype):
        return partial(self.__call__, obj)


if __name__ == '__main__':
    import time

    def fibonacci(n):
        """fibonacci docstring"""
        if n in (0, 1):
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    fibonacci_with_memo = Memoize(fibonacci)
    print('#### Check the info of the wrapped ####')
    print(f'Docstring: {fibonacci_with_memo.__doc__}')
    print(f'Function-name: {fibonacci_with_memo.__name__}')
    print(f'Repr: {fibonacci_with_memo}')

    print('#### Without memo ####')
    t = time.perf_counter()
    fibonacci(20)
    print(f'First: {time.perf_counter() - t}')

    t = time.perf_counter()
    fibonacci(20)
    print(f'Second: {time.perf_counter() - t}')

    print('#### With memo ####')
    t = time.perf_counter()
    fibonacci_with_memo(20)
    print(f'First: {time.perf_counter() - t}')

    t = time.perf_counter()
    fibonacci_with_memo(20)
    print(f'Second: {time.perf_counter() - t}')
