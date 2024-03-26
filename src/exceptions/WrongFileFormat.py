class WrongFileFormatException(Exception):
    """
    Exception class for indicating an invalid file format.

    Attributes:
        message (str): Error message indicating the wrong file format.
    """
    def __init__(self, filename: str):
        """
        Initializes the WrongFileFormatException.

        Args:
            filename (str): Name of the file with the wrong format.

        Returns:
            None
        """
        self.message = f'Wrong format for the file: {filename}'
        super(WrongFileFormatException, self).__init__(self.message)
