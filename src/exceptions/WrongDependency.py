class WrongDependencyException(Exception):
    pass

    def __init__(self, fltr: str, arguments: str):
        self.message = "Wrong dependency for " + fltr + ": " + str(arguments)
        super(WrongDependencyException, self).__init__(self.message)
