class WrongFilterNameException(Exception):
    pass

    def __init__(self, fltr: str):
        self.message = "Wrong filter name: " + fltr
        super(WrongFilterNameException, self).__init__(self.message)
