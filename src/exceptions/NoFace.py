class NoFaceException(Exception):
    pass

    def __init__(self, arguments: str):
        self.message = "No face on the photo: " + str(arguments)
        super(NoFaceException, self).__init__(self.message)
