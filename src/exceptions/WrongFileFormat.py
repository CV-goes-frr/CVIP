class WrongFileFormatException(Exception):
    def __init__(self, filename: str):
        self.message = f'Wrong format for the file: {filename}'
        super(WrongFileFormatException, self).__init__(self.message)
