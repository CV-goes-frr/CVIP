class NoFaceException(Exception):
    """
    Exception class for indicating no face detected in the photo.

    Attributes:
        message (str): Error message indicating the absence of a face.
    """
    def __init__(self, arguments: str):
        """
        Initializes the NoFaceException.

        Args:
            arguments (str): Description of the issue.

        Returns:
            None
        """
        self.message = "No face on the photo: " + str(arguments)
        super(NoFaceException, self).__init__(self.message)
