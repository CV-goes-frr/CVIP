import os
import errno
import re

from typing import List

from .Processor import Processor
from .VerifyArgs import VerifyArgs
from .exceptions.WrongDependency import WrongDependencyException


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

            # check filter name and connected arguments
            check_args: VerifyArgs = VerifyArgs(command[1])
            check_args.check()

            # make dependencies
            res_obj.label_dependencies[command[2]] = []
            for c in command[0].split(':'):
                res_obj.label_dependencies[command[2]].append(c)

            # create filter objects with parameters
            for label in command[2].split(":"):
                print(command[1][1:])
                res_obj.label_in_map[label] = res_obj.class_map[command[1][0]](*(command[1][1:]))

            # create dependencies for each out_label and update calls_counters
            for out_label in command[2].split(':'):
                res_obj.label_dependencies[out_label] = []
                if final:
                    res_obj.label_in_map[out_label].calls_counter += 1

                for in_label in command[0].split(':'):
                    if in_label not in res_obj.label_in_map and in_label[0:3] != '-i=':
                        raise WrongDependencyException(command[1][0], in_label)

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
