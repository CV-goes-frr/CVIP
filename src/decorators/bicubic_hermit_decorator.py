from functools import lru_cache, wraps

import numpy as np


def bicubic_hermit_cache(function):
    """
    Overriding wrapper function to make input data hashable and
    cached_wrapper function to call the original function with hashable input data.

    :param function: - original function
    :return: - wrapper function
    """
    @lru_cache(maxsize=128)
    def cached_wrapper(hashable_array1: tuple, hashable_array2: tuple,
                       hashable_array3: tuple, hashable_array4: tuple, t: int):
        # cast tuples to np.array(s)
        array1 = np.array(hashable_array1)
        array2 = np.array(hashable_array2)
        array3 = np.array(hashable_array3)
        array4 = np.array(hashable_array4)
        return function(array1, array2, array3, array4, t)

    @wraps(function)
    def wrapper(array1: np.ndarray, array2: np.ndarray,
                array3: np.ndarray, array4: np.ndarray, t: int):
        # cast np.ndarray(s) to tuples
        return cached_wrapper(tuple(array1), tuple(array2), tuple(array3), tuple(array4), t)

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper
