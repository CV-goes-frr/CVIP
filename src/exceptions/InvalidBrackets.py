class InvalidBracketsException(Exception):
    """
    Exception class for indicating invalid brackets in a prompt.

    Attributes:
        message (str): Error message indicating the issue with brackets.
    """
    def __init__(self, msg: str):
        """
        Initializes the InvalidBracketsException.

        Args:
            msg (str): The error message describing the issue with brackets.

        Returns:
            None
        """
        self.message = f'Your prompt is invalid in terms of brackets: {msg}'
        super(InvalidBracketsException, self).__init__(self.message)
