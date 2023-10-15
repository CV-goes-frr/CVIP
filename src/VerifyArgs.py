from .exceptions.WrongParameters import WrongParametersException
from .exceptions.WrongFiltername import WrongFilterNameException


class VerifyArgs:

    def __init__(self, args: str):
        self.args: str = args
        self.name: str = args[0]

    def check(self):
        match self.name:
            case 'crop':
                if len(self.args) != 5:
                    raise WrongParametersException(self.name, str(self.args[1:]))
            case 'nn_scale':
                if len(self.args) != 2:
                    raise WrongParametersException(self.name, str(self.args[1:]))
            case 'bilinear_scale':
                if len(self.args) != 2:
                    raise WrongParametersException(self.name, str(self.args[1:]))
            case 'bicubic_scale':
                if len(self.args) != 2:
                    raise WrongParametersException(self.name, str(self.args[1:]))
            case 'merge':
                if len(self.args) != 1:
                    raise WrongParametersException(self.name, str(self.args[1:]))
            case 'face_blur':
                if len(self.args) != 2:
                    raise WrongParametersException(self.name, str(self.args[1:]))
            case 'face_detection':
                if len(self.args) != 1:
                    raise WrongParametersException(self.name, str(self.args[1:]))
            case 'duplicate':
                pass
            case _:
                raise WrongFilterNameException(self.name)
