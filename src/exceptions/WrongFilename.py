class WrongFilenameException(Exception):
    """
    Exception class for indicating an invalid filename containing forbidden symbols.

    Attributes:
        message (str): Error message indicating the invalid filename.
    """
    def __init__(self, filename: str):
        """
        Initializes the WrongFilenameException.

        Args:
            filename (str): Name of the invalid filename.

        Returns:
            None
        """
        self.message = f'Wrong filename (contains forbidden symbols): {filename}'
        super(WrongFilenameException, self).__init__(self.message)
