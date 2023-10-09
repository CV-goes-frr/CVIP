import os
import errno
import re

from typing import List

from .Processor import Processor


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
        inp_commands: List[List[any]] = []
        for i in range(len(inp_parameters)):
            inp_commands.append(re.split('\[|\]', inp_parameters[i]))

        for command in inp_commands:
            command[1] = command[1].split(':')
            final: bool = False

            # add label if we want output file there
            if command[2][0:3] == '-o=':
                final = True
                command[2] = command[2][3::]
                for label in command[2].split(':'):
                    res_obj.fin_labels.append(label)

            match command[1][0]:
                case 'crop':
                    if len(command[1]) != 5:
                        raise Exception("Wrong number of parameters for crop")
                    res_obj.label_in_map[command[2]] = res_obj.class_map["crop"](command[1][1],
                                                                                 command[1][2],
                                                                                 command[1][3],
                                                                                 command[1][4])
                case 'nn_scale':
                    if len(command[1]) != 2:
                        raise Exception("Wrong number of parameters for nn_scale")
                    res_obj.label_in_map[command[2]] = res_obj.class_map["nn_scale"](command[1][1])
                case 'bilinear_scale':
                    if len(command[1]) != 2:
                        raise Exception("Wrong number of parameters for bilinear_scale")
                    res_obj.label_in_map[command[2]] = res_obj.class_map["bilinear_scale"](command[1][1])
                case 'bicubic_scale':
                    if len(command[1]) != 2:
                        raise Exception("Wrong number of parameters for bicubic_scale")
                    res_obj.label_in_map[command[2]] = res_obj.class_map["bicubic_scale"](command[1][1])
                case 'merge':
                    if len(command[1]) != 1:
                        raise Exception("Wrong number of parameters for merge")

                    res_obj.label_in_map[command[2]] = res_obj.class_map["merge"]()
                case 'duplicate':
                    if len(command[1]) != 1:
                        raise Exception("Wrong number of parameters for duplicate")
                    obj_in_map = res_obj.class_map["duplicate"]()
                    obj_in_map.return_all = False
                    res_obj.label_in_map[command[2].split(':')[0]] = obj_in_map
                    res_obj.label_in_map[command[2].split(':')[1]] = obj_in_map
                case _:
                    raise Exception("Wrong filter name: " + command[1][0])

            # create dependencies for each out_label and update calls_counters
            for out_label in command[2].split(':'):
                res_obj.label_dependencies[out_label] = []
                if final:
                    res_obj.label_in_map[out_label].calls_counter += 1

                for in_label in command[0].split(':'):
                    if in_label not in res_obj.label_in_map and in_label != '-i':
                        raise Exception("Dependency label for " + command[1][0] +
                                        " doesn't exist at this moment: " + in_label)

                    res_obj.label_dependencies[out_label].append(in_label)
                    if in_label != '-i':
                        res_obj.label_in_map[in_label].calls_counter += 1

                res_obj.labels_to_out[out_label] = command[2].split(':')

        print("\nDependencies:")
        for key in res_obj.label_dependencies:
            print(key, res_obj.label_dependencies[key])

        print("\nLabels mapping to filter objects:")
        for key in res_obj.label_in_map:
            print(key, res_obj.label_in_map[key], "calls:", res_obj.label_in_map[key].calls_counter)

        print("\nLabels to out:")
        for key in res_obj.labels_to_out:
            print(key, res_obj.labels_to_out[key])

        # find final labels with 0 calls
        # for key in res_obj.label_in_map:
        #     if res_obj.label_in_map[key].calls_counter == 0:
        #         res_obj.fin_labels.append(key)

        print("\nFinal labels:")
        for label in res_obj.fin_labels:
            print(label)

        if not os.path.exists(res_obj.inp_image):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), res_obj.inp_image)

        print()
        return res_obj
