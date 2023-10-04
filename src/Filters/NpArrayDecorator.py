import numpy as np


def np_arr_to_tuple(function):
    def wrapper(*args):
        # args = [tuple([tuple(inner_x) for inner_x in x])
        #         if type(x) == np.ndarray else x for x in args]
        #
        # result = function(tuple(args))
        # result = [tuple([tuple(inner_x) for inner_x in x])
        #           if type(x) == np.ndarray else x for x in result]
        # return result
        args = [x.tostring() if type(x) == np.ndarray else x for x in args]
        result = function(*args)
        return result

    return wrapper
