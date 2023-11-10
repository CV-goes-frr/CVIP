import re

from src.exceptions.WrongParameters import WrongParametersException
from src.exceptions.WrongFiltername import WrongFilterNameException

class VerifyArgs:

    def __init__(self, args: str):
        self.args: str = args
        self.name: str = args[0]

    def check(self):
        # Check the filter name and its parameters based on the name
        match self.name:
            case 'crop':
                if len(self.args) != 5:
                    raise WrongParametersException(self.name, str(self.args[1:]))
                # Validate that all parameters are numeric
                for param in self.args[1:]:
                    if not re.match(r'^[0-9]+$', param):
                        raise WrongParametersException(self.name, param)

            case 'nn_scale':
                if len(self.args) != 2:
                    raise WrongParametersException(self.name, str(self.args[1:]))
                # Validate that all parameters are numeric
                for param in self.args[1:]:
                    if not re.match(r'^[0-9]+$', param):
                        raise WrongParametersException(self.name, param)

            case 'bilinear_scale':
                if len(self.args) != 2:
                    raise WrongParametersException(self.name, str(self.args[1:]))
                # Validate that all parameters are numeric
                for param in self.args[1:]:
                    if not re.match(r'^[0-9]+$', param):
                        raise WrongParametersException(self.name, param)

            case 'bicubic_scale':
                if len(self.args) != 2:
                    raise WrongParametersException(self.name, str(self.args[1:]))
                # Validate that all parameters are numeric
                for param in self.args[1:5]:
                    if not re.match(r'^[0-9]+$', param):
                        raise WrongParametersException(self.name, param)

            case 'merge':
                if len(self.args) != 1:
                    raise WrongParametersException(self.name, str(self.args[1:]))

            case 'face_blur':
                if len(self.args) != 2:
                    raise WrongParametersException(self.name, str(self.args[1:]))
                # Validate that all parameters are numeric
                for param in self.args[1:5]:
                    if not re.match(r'^[0-9]+$', param):
                        raise WrongParametersException(self.name, param)

            case 'face_detection':
                if len(self.args) != 1:
                    raise WrongParametersException(self.name, str(self.args[1:]))

            case 'duplicate':
                pass

            case _:
                # If the filter name is not recognized, raise an exception
                raise WrongFilterNameException(self.name)
