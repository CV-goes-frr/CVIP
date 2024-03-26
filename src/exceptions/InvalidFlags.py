class InvalidFlagsException(Exception):
    """
    Exception class for indicating invalid flags.

    Attributes:
        message (str): Error message indicating the invalid flag.
    """
    def __init__(self, flag: str):
        """
        Initializes the InvalidFlagsException.

        Args:
            flag (str): The invalid flag encountered.

        Returns:
            None
        """
        self.message = f'Invalid flags: {flag}'
        super(InvalidFlagsException, self).__init__(self.message)
