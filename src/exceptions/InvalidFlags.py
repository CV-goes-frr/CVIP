class InvalidFlagsException(Exception):
    def __init__(self, flag: str):
        self.message = f'Invalid flags: {flag}'
        super(InvalidFlagsException, self).__init__(self.message)
