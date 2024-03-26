from functools import lru_cache, wraps

import numpy as np


def bilinear_weight_cache(function):
    """
    Decorator for caching the results of a function that computes bilinear interpolation weights.

    Args:
        function: Original function to be decorated.

    Returns:
        function: Wrapper function with caching functionality.
    """
    @lru_cache(maxsize=128)
    def cached_wrapper(alpha: int, beta: int, top_left: tuple, top_right: tuple,
                       bottom_left: tuple, bottom_right: tuple):
        """
        Wrapper function that caches the results of the original function based on hashable input data.

        Args:
            alpha (int): Interpolation factor for x-direction, ranging from 0 to 1.
            beta (int): Interpolation factor for y-direction, ranging from 0 to 1.
            top_left (tuple): Tuple representing the pixel value of the top-left corner.
            top_right (tuple): Tuple representing the pixel value of the top-right corner.
            bottom_left (tuple): Tuple representing the pixel value of the bottom-left corner.
            bottom_right (tuple): Tuple representing the pixel value of the bottom-right corner.

        Returns:
            result: Result of calling the original function with the corresponding parameters.

        """
        # Cast tuples to np.arrays
        return function(alpha, beta, np.array(top_left), np.array(top_right),
                        np.array(bottom_left), np.array(bottom_right))

    @wraps(function)
    def wrapper(alpha: int, beta: int, top_left: np.ndarray, top_right: np.ndarray,
                bottom_left: np.ndarray, bottom_right: np.ndarray):
        """
        Wrapper function that casts numpy arrays to tuples before calling the cached wrapper.

        Args:
            alpha (int): Interpolation factor for x-direction, ranging from 0 to 1.
            beta (int): Interpolation factor for y-direction, ranging from 0 to 1.
            top_left (np.ndarray): Pixel value of the top-left corner.
            top_right (np.ndarray): Pixel value of the top-right corner.
            bottom_left (np.ndarray): Pixel value of the bottom-left corner.
            bottom_right (np.ndarray): Pixel value of the bottom-right corner.

        Returns:
            result: Result of calling the cached wrapper function with the corresponding parameters.
        """
        # Cast np.ndarray(s) to tuples
        return cached_wrapper(alpha, beta, tuple(top_left), tuple(top_right),
                              tuple(bottom_left), tuple(bottom_right))

    # Copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper
