import os
import errno
import re

from typing import List

from src.Processor import Processor
from src.VerifyArgs import VerifyArgs


class Parser:

    def __init__(self, inp_actions: str, processes_limit: str):
        self.inp_actions: str = inp_actions
        self.processes_limit: int = int(processes_limit)

    def parse(self) -> Processor:
        res_obj: Processor = Processor(self.processes_limit)

        inp_parameters: List[str] = self.inp_actions.split('][')
        inp_parameters[0] = inp_parameters[0][1:]  # delete first '['
        inp_parameters[-1] = inp_parameters[-1][:-1]  # delete last ']'
        for i in range(len(inp_parameters)):
            inp_parameters[i] = re.split('\[|\]', inp_parameters[i])

        for command in inp_parameters:
            command[1] = command[1].split(':')
            final: bool = False

            # add label if we want output file there
            if command[2][0:3] == '-o=':
                final = True
                command[2] = command[2][3::]
                for label in command[2].split(':'):
                    res_obj.fin_labels.append(label)

            check_args: VerifyArgs = VerifyArgs(command[1])
            check_args.check()

            match command[1][0]:
                case 'crop':
                    res_obj.label_dependencies[command[2]] = [command[0]]
                    res_obj.label_in_map[command[2]] = res_obj.class_map["crop"](command[1][1],
                                                                                 command[1][2],
                                                                                 command[1][3],
                                                                                 command[1][4],
                                                                                )
                case 'nn_scale':
                    res_obj.label_dependencies[command[2]] = [command[0]]
                    res_obj.label_in_map[command[2]] = res_obj.class_map["nn_scale"](command[1][1])
                case 'bilinear_scale':
                    res_obj.label_dependencies[command[2]] = [command[0]]
                    res_obj.label_in_map[command[2]] = res_obj.class_map["bilinear_scale"](command[1][1])
                case 'bicubic_scale':
                    res_obj.label_dependencies[command[2]] = [command[0]]
                    res_obj.label_in_map[command[2]] = res_obj.class_map["bicubic_scale"](command[1][1])
                case 'merge':
                    res_obj.label_dependencies[command[2]] = []
                    for c in command[0].split(':'):
                        res_obj.label_dependencies[command[2]].append(c)
                    res_obj.label_in_map[command[2]] = res_obj.class_map["merge"]()
                case 'duplicate':
                    if len(command[1]) != 1:
                        raise Exception("Wrong number of parameters for duplicate")
                    obj_in_map = res_obj.class_map["duplicate"]()
                    obj_in_map.return_all = False
                    res_obj.label_in_map[command[2].split(':')[0]] = obj_in_map
                    res_obj.label_in_map[command[2].split(':')[1]] = obj_in_map
                case 'face_blur':
                    res_obj.label_dependencies[command[2]] = [command[0]]
                    print(command[1])
                    res_obj.label_in_map[command[2]] = res_obj.class_map["face_blur"](command[1][1])
                case _:
                    raise Exception("Wrong filter name: " + command[1][0])

            # create dependencies for each out_label and update calls_counters
            for out_label in command[2].split(':'):
                res_obj.label_dependencies[out_label] = []
                if final:
                    res_obj.label_in_map[out_label].calls_counter += 1

                for in_label in command[0].split(':'):
                    if in_label not in res_obj.label_in_map and in_label[0:3] != '-i=':
                        raise Exception("Dependency label for " + command[1][0] +
                                        " doesn't exist at this moment: " + in_label)

                    res_obj.label_dependencies[out_label].append(in_label)
                    if in_label[0:3] != '-i=':
                        res_obj.label_in_map[in_label].calls_counter += 1

                res_obj.labels_to_out[out_label] = command[2].split(':')

        print("\nDependencies:")
        for key in res_obj.label_dependencies:
            print(key, res_obj.label_dependencies[key])

        print("\nLabels map to filters:")
        for key in res_obj.label_in_map:
            print(key, res_obj.label_in_map[key], "calls:", res_obj.label_in_map[key].calls_counter)

        print("\nLabels to out:")
        for key in res_obj.labels_to_out:
            print(key, res_obj.labels_to_out[key])

        print("\nFinal labels:")
        for label in res_obj.fin_labels:
            print(label)

        for key in res_obj.label_dependencies:
            if res_obj.label_dependencies[key][0][0:3] == '-i=' and \
                    not os.path.exists(res_obj.label_dependencies[key][0][3::]):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        res_obj.label_dependencies[key][0][3::])

        print()
        return res_obj
