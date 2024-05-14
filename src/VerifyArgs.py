import re

from .exceptions.WrongFiltername import WrongFilterNameException
from .exceptions.WrongParameters import WrongParametersException


class VerifyArgs:

    def __init__(self, args: str):
        """
        Initializes VerifyArgs object with args and its name.

        Args:
            args (str): Arguments string.
        """
        self.args: str = args
        self.name: str = args[0]

    def check(self):
        """
        Checks the filter name and its parameters based on the name.
        Raises exceptions for wrong parameters or filter name.
        """
        # Check the filter name and its parameters based on the name
        match self.name:
            case 'crop':
                if len(self.args) != 5:
                    raise WrongParametersException(self.name, str(self.args[1:]))
                # Validate that all parameters are numeric
                for param in self.args[1:]:
                    if not re.match(r'^[0-9]+$', param):
                        raise WrongParametersException(self.name, param)

            case 'nn_scale_with_factor':
                if len(self.args) != 2:
                    raise WrongParametersException(self.name, str(self.args[1:]))
                # Validate that all parameters are numeric
                for param in self.args[1:]:
                    if not re.match(r'^[0-9]+$', param):
                        raise WrongParametersException(self.name, param)

            case 'bilinear_scale_with_factor':
                if len(self.args) != 2:
                    raise WrongParametersException(self.name, str(self.args[1:]))
                # Validate that all parameters are numeric
                for param in self.args[1:]:
                    if not re.match(r'^[0-9]+$', param):
                        raise WrongParametersException(self.name, param)

            case 'scale_to_resolution':
                if len(self.args) != 3:
                    raise WrongParametersException(self.name, str(self.args[1:]))
                # Validate that all parameters are numeric
                for param in self.args[1:5]:
                    if not re.match(r'^[0-9]+$', param):
                        raise WrongParametersException(self.name, param)

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

            case 'mask':
                if len(self.args) != 2:
                    raise WrongParametersException(self.name, str(self.args[1:]))

            case 'feature_matching':
                if len(self.args) != 3:
                    raise WrongParametersException(self.name, str(self.args[1:]))

            case 'motion_tracking':
                if len(self.args) != 1:
                    raise WrongParametersException(self.name, str(self.args[1:]))

            case 'panorama':
                if len(self.args) != 1:
                    raise WrongParametersException(self.name, str(self.args[1:]))

            case 'video_overlay':
                if len(self.args) != 6:
                    raise WrongParametersException(self.name, str(self.args[1:]))
                for param in self.args[2:]:
                    if not re.match(r'^[0-9]+$', param):
                        raise WrongParametersException(self.name, param)
                if int(self.args[2]) < 1:
                    raise WrongParametersException(self.name, self.args[2])
                if int(self.args[5]) != 0 and int(self.args[5]) != 1:
                    raise WrongParametersException(self.name, self.args[5])

            case _:
                # If the filter name is not recognized, raise an exception
                raise WrongFilterNameException(self.name)
