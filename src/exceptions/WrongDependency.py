class WrongDependencyException(Exception):
    """
    Exception class for indicating an invalid dependency.

    Attributes:
        message (str): Error message indicating the wrong dependency.
    """
    def __init__(self, fltr: str, arguments: str):
        """
        Initializes the WrongDependencyException.

        Args:
            fltr (str): Name of the filter with the wrong dependency.
            arguments (str): Description of the wrong dependency.

        Returns:
            None
        """
        self.message = "Wrong dependency for " + fltr + ": " + str(arguments)
        super(WrongDependencyException, self).__init__(self.message)
