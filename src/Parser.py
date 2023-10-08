import os
import errno
import re

from typing import List

from src.Processor import Processor


class Parser:

    def __init__(self, inp_file: str, inp_actions: str, processes_limit: str):
        self.inp_image: str = inp_file
        self.inp_actions: str = inp_actions
        self.processes_limit: int = int(processes_limit)

    def parse(self) -> Processor:
        res_obj: Processor = Processor(self.processes_limit)
        res_obj.inp_image = self.inp_image

        inp_parameters: List[str] = self.inp_actions.split('][')
        inp_parameters[0] = inp_parameters[0][1:]  # delete first '['
        inp_parameters[-1] = inp_parameters[-1][:-1]  # delete last ']'
        for i in range(len(inp_parameters)):
            inp_parameters[i] = re.split('\[|\]', inp_parameters[i])

        for command in inp_parameters:
            command[1] = command[1].split(':')
            match command[1][0]:
                case 'crop':
                    if len(command[1]) != 5:
                        raise Exception("Wrong number of parameters for crop")
                    res_obj.label_dependencies[command[2]] = [command[0]]
                    res_obj.label_in_map[command[2]] = res_obj.class_map["crop"](command[1][1],
                                                                                 command[1][2],
                                                                                 command[1][3],
                                                                                 command[1][4])
                case 'nn_scale':
                    if len(command[1]) != 2:
                        raise Exception("Wrong number of parameters for nn_scale")
                    res_obj.label_dependencies[command[2]] = [command[0]]

                    res_obj.label_in_map[command[2]] = res_obj.class_map["nn_scale"](command[1][1])
                case 'bilinear_scale':
                    if len(command[1]) != 2:
                        raise Exception("Wrong number of parameters for bilinear_scale")
                    res_obj.label_dependencies[command[2]] = [command[0]]
                    res_obj.label_in_map[command[2]] = res_obj.class_map["bilinear_scale"](command[1][1])
                case 'bicubic_scale':
                    if len(command[1]) != 2:
                        raise Exception("Wrong number of parameters for bicubic_scale")
                    res_obj.label_dependencies[command[2]] = [command[0]]
                    res_obj.label_in_map[command[2]] = res_obj.class_map["bicubic_scale"](command[1][1])
                case 'merge':
                    if len(command[1]) != 1:
                        raise Exception("Wrong number of parameters for merge")
                    res_obj.label_dependencies[command[2]] = []
                    for c in command[0].split(':'):
                        res_obj.label_dependencies[command[2]].append(c)
                    res_obj.label_in_map[command[2]] = res_obj.class_map["merge"]()

                case 'face_blur':
                    if len(command[1]) != 2:
                        raise Exception("Wrong number of parameters for merge")
                    res_obj.label_dependencies[command[2]] = [command[0]]
                    res_obj.label_in_map[command[2]] = res_obj.class_map["face_blur"](command[1][1])

                case _:
                    raise Exception("Wrong filter name: " + command[1][0])

            # increase calls counter
            for c in command[0].split(':'):
                if c in res_obj.label_in_map:
                    res_obj.label_in_map[c].calls_counter += 1

        print("\nDependencies:")
        for key in res_obj.label_dependencies:
            print(key, res_obj.label_dependencies[key])

        print("\nLabels map to filters:")
        for key in res_obj.label_in_map:
            print(key, res_obj.label_in_map[key], "calls:", res_obj.label_in_map[key].calls_counter)

        res_obj.fin = inp_parameters[-1][-1]

        if not os.path.exists(res_obj.inp_image):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), res_obj.inp_image)

        print()
        return res_obj