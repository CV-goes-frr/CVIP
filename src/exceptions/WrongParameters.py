class WrongParametersException(Exception):
    """
    Exception class for indicating wrong parameters passed to a filter.

    Attributes:
        message (str): Error message indicating the filter name and the wrong parameters.
    """
    def __init__(self, fltr: str, arguments: str):
        """
        Initializes the WrongParametersException.

        Args:
            fltr (str): Name of the filter.
            arguments (str): Wrong parameters passed to the filter.

        Returns:
            None
        """
        self.message = "Wrong parameters for " + str(fltr) + ": " + str(arguments)
        super(WrongParametersException, self).__init__(self.message)
