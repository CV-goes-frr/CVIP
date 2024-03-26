import errno
import os
import re
from typing import List

from settings import prefix
from .Processor import Processor
from .VerifyArgs import VerifyArgs
from .exceptions.WrongDependency import WrongDependencyException


class Parser:

    def __init__(self, inp_actions: str, processes_limit: str):
        """
        Initializes Parser object with input actions and processes limit.

        Args:
            inp_actions (str): Pipeline string.
            processes_limit (str): Number of processes in the pool.
        """
        self.inp_actions: str = inp_actions
        self.processes_limit: int = int(processes_limit)

    def parse(self, video_editing: bool) -> Processor:
        """
        Parses the input actions and constructs a Processor object.

        Args:
            video_editing (bool): Flag indicating video editing mode.

        Returns:
            Processor: A Processor object.
        """
        res_obj: Processor = Processor(self.processes_limit, video_editing)

        inp_parameters: List[str] = self.inp_actions.split('][')
        inp_parameters[0] = inp_parameters[0][1:]
        inp_parameters[-1] = inp_parameters[-1][:-1]

        for i in range(len(inp_parameters)):
            inp_parameters[i] = re.split('\[|\]', inp_parameters[i])

        for command in inp_parameters:
            command[1] = command[1].split(':')

            # check filter name and connected arguments
            check_args: VerifyArgs = VerifyArgs(command[1])
            check_args.check()

            # create filter objects with parameters
            for label in command[2].split(":"):
                if label[0:3] != '-o=':
                    res_obj.label_in_map[label] = res_obj.class_map[command[1][0]](*(command[1][1:]))
                else:
                    res_obj.label_in_map[label[3::]] = res_obj.class_map[command[1][0]](*(command[1][1:]))

            # specify dependencies for every filter
            inp_labels_splited = command[2].split(':')
            for ind in range(len(inp_labels_splited)):
                if inp_labels_splited[ind][0:3] == '-o=':
                    inp_labels_splited[ind] = inp_labels_splited[ind][3::]
                    res_obj.label_in_map[inp_labels_splited[ind]].calls_counter += 1
                    res_obj.fin_labels.append(inp_labels_splited[ind])

                res_obj.label_dependencies[inp_labels_splited[ind]] = []
                for in_label in command[0].split(':'):
                    if in_label not in res_obj.label_in_map and in_label[0:3] != '-i=':
                        raise WrongDependencyException(command[1][0], in_label)
                    res_obj.label_dependencies[inp_labels_splited[ind]].append(in_label)
                    if in_label[0:3] != '-i=':
                        res_obj.label_in_map[in_label].calls_counter += 1

                res_obj.labels_to_out[inp_labels_splited[ind]] = inp_labels_splited

        print("\nFiles with the following names will be created:")
        for label in res_obj.fin_labels:
            print(label)

        # check if input files are incorrect
        for key in res_obj.label_dependencies:
            if res_obj.label_dependencies[key][0][0:3] == '-i=' and \
                    not os.path.exists(prefix + '/' + res_obj.label_dependencies[key][0][3::]):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        prefix + '/' + res_obj.label_dependencies[key][0][3::])

        print()
        return res_obj
