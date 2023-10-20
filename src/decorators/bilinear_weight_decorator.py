from functools import lru_cache, wraps

import numpy as np


def bilinear_weight_cache(function):
    @lru_cache(maxsize=128)
    def cached_wrapper(alpha: int, beta: int, top_left: tuple, top_right: tuple,
                       bottom_left: tuple, bottom_right: tuple):
        # cast tuples to np.array(s)
        return function(alpha, beta, np.array(top_left), np.array(top_right),
                        np.array(bottom_left), np.array(bottom_right))

    @wraps(function)
    def wrapper(alpha: int, beta: int, top_left: np.ndarray, top_right: np.ndarray,
                bottom_left: np.ndarray, bottom_right: np.ndarray):
        # cast np.ndarray(s) to tuples
        return cached_wrapper(alpha, beta, tuple(top_left), tuple(top_right),
                              tuple(bottom_left), tuple(bottom_right))

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper
