import sys
import os
import errno

import numpy as np
import re

from Maintenance.Processor import Processor


class Parser:

    def __init__(self, inp_file: str, inp_actions: str):
        self.inp_image = inp_file
        self.inp_actions = inp_actions

    def parse(self) -> Processor:
        res_obj = Processor()
        res_obj.inp_image = self.inp_image

        inp_parameters = self.inp_actions.split('][')
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
                        raise Exception("Wrong number of parameters for nn_scale")
                    res_obj.label_dependencies[command[2]] = [command[0]]
                    res_obj.label_in_map[command[2]] = res_obj.class_map["bilinear_scale"](command[1][1])
                case 'merge':
                    if len(command[1]) != 1:
                        raise Exception("Wrong number of parameters for merge")
                    res_obj.label_dependencies[command[2]] = []
                    for c in command[0].split(':'):
                        res_obj.label_dependencies[command[2]].append(c)
                    res_obj.label_in_map[command[2]] = res_obj.class_map["merge"]()

                case _:
                    raise Exception("Wrong filter name: " + command[1][0])
        
        print("\nDependencies:")
        for key in res_obj.label_dependencies:
            print(key, res_obj.label_dependencies[key])
        
        print("\nLabels map to filters:")
        for key in res_obj.label_in_map:
            print(key, res_obj.label_in_map[key])
            
        res_obj.fin = inp_parameters[-1][-1]

        if not os.path.exists(res_obj.inp_image):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), res_obj.inp_image)

        return res_obj
