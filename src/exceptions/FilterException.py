class FilterException(Exception):
    def __init__(self, message, foo, *args):
        self.message = message  # without this you may get DeprecationWarning
        # Special attribute you desire with your Error,
        # perhaps the value that caused the error?:
        self.foo = foo
        # allow users initialize misc. arguments as any other builtin Error
        super(FilterException, self).__init__(message, foo, *args)
