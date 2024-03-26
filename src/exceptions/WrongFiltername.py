class WrongFilterNameException(Exception):
    """
    Exception class for indicating an invalid filter name.

    Attributes:
        message (str): Error message indicating the invalid filter name.
    """
    def __init__(self, fltr: str):
        """
        Initializes the WrongFilterNameException.

        Args:
            fltr (str): Name of the invalid filter.

        Returns:
            None
        """
        self.message = "Wrong filter name: " + fltr
        super(WrongFilterNameException, self).__init__(self.message)
