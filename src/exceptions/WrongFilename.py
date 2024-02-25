class WrongFilenameException(Exception):
    def __init__(self, filename: str):
        self.message = f'Wrong filename (contains forbidden symbols): {filename}'
        super(WrongFilenameException, self).__init__(self.message)
