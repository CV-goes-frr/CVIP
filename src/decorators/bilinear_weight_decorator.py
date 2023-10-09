from functools import lru_cache, wraps

import numpy as np


def bilinear_weight_cache(function):
    @lru_cache(maxsize=128)
    def cached_wrapper(alpha: int, beta: int, top_left, top_right, bottom_left, bottom_right):
        return function(alpha, beta, np.array(top_left), np.array(top_right),
                        np.array(bottom_left), np.array(bottom_right))

    @wraps(function)
    def wrapper(alpha: int, beta: int, top_left, top_right, bottom_left, bottom_right):
        return cached_wrapper(alpha, beta, tuple(top_left), tuple(top_right),
                              tuple(bottom_left), tuple(bottom_right))

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper
