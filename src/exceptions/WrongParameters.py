class WrongParametersException(Exception):
    pass

    def __init__(self, fltr: str, arguments: str):
        self.message = "Wrong parameters for " + fltr + ": " + str(arguments)
        super(WrongParametersException, self).__init__(self.message)
