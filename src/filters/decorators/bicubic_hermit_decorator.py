from functools import lru_cache, wraps

import numpy as np


def bicubic_hermit_cache(function):
    """
    Decorator for caching the results of a function that accepts numpy arrays as input parameters.

    Args:
        function: Original function to be decorated.

    Returns:
        function: Wrapper function with caching functionality.
    """

    @lru_cache(maxsize=128)
    def cached_wrapper(hashable_array1: tuple, hashable_array2: tuple,
                       hashable_array3: tuple, hashable_array4: tuple, t: int):
        """
        Wrapper function that caches the results of the original function based on hashable input data.

        Args:
            hashable_array1 (tuple): Hashable tuple representing the first numpy array.
            hashable_array2 (tuple): Hashable tuple representing the second numpy array.
            hashable_array3 (tuple): Hashable tuple representing the third numpy array.
            hashable_array4 (tuple): Hashable tuple representing the fourth numpy array.
            t (int): Integer parameter.

        Returns:
            result: Result of calling the original function with the corresponding numpy arrays.

        """
        # Cast tuples to np.arrays
        array1 = np.array(hashable_array1)
        array2 = np.array(hashable_array2)
        array3 = np.array(hashable_array3)
        array4 = np.array(hashable_array4)
        return function(array1, array2, array3, array4, t)

    @wraps(function)
    def wrapper(array1: np.ndarray, array2: np.ndarray,
                array3: np.ndarray, array4: np.ndarray, t: int):
        """
        Wrapper function that casts numpy arrays to tuples before calling the cached wrapper.

        Args:
            array1 (np.ndarray): First numpy array.
            array2 (np.ndarray): Second numpy array.
            array3 (np.ndarray): Third numpy array.
            array4 (np.ndarray): Fourth numpy array.
            t (int): Integer parameter.

        Returns:
            result: Result of calling the cached wrapper function with the corresponding numpy arrays.
        """
        # Cast np.ndarray(s) to tuples
        return cached_wrapper(tuple(array1), tuple(array2), tuple(array3), tuple(array4), t)

    # Copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper
