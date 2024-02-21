class InvalidBracketsException(Exception):
    def __init__(self, msg: str):
        self.message = f'Your prompt is invalid in terms of brackets: {msg}'
        super(InvalidBracketsException, self).__init__(self.message)
