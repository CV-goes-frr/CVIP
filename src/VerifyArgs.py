class VerifyArgs:

    def __init__(self, args: str):
        self.args: str = args
        self.name: str = args[0]

    def check(self):
        match self.name:
            case 'crop':
                if len(self.args) != 5:
                    raise Exception("Wrong number of parameters for crop")
            case 'nn_scale':
                if len(self.args) != 2:
                    raise Exception("Wrong number of parameters for nn_scale")
            case 'bilinear_scale':
                if len(self.args) != 2:
                    raise Exception("Wrong number of parameters for bilinear_scale")
            case 'bicubic_scale':
                if len(self.args) != 2:
                    raise Exception("Wrong number of parameters for bicubic_scale")
            case 'merge':
                if len(self.args) != 1:
                    raise Exception("Wrong number of parameters for merge")
            case 'face_blur':
                if len(self.args) != 2:
                    raise Exception("Wrong number of parameters for face_blur")
            case _:
                raise Exception("Wrong filter name: " + self.name)

