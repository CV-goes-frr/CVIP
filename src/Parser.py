import os
import errno
import re

from typing import List

from .Processor import Processor
from .VerifyArgs import VerifyArgs
from src.exceptions.WrongDependency import WrongDependencyException
from settings import prefix


class Parser:

    def __init__(self, inp_actions: str, processes_limit: str):
        """
        :param inp_actions: pipeline
        :param processes_limit: what number of processes should be in our pool
        """
        self.inp_actions: str = inp_actions
        self.processes_limit: int = int(processes_limit)

    def parse(self) -> Processor:
        """
        Parser collects everything into one working structure - Processor.

        :return: Processor object
        """

        res_obj: Processor = Processor(self.processes_limit)

        inp_parameters: List[str] = self.inp_actions.split('][')  # split the whole line into filters with in/out labels
        inp_parameters[0] = inp_parameters[0][1:]  # delete first '[' and
        inp_parameters[-1] = inp_parameters[-1][:-1]  # delete last ']' that weren't deleted by split

        # split the line like "in-label1:in-label2]filter_name:param1:param2[out-label1:out-label2"
        # into ['in-label1:in-label2', 'filter_name:param1:param2', 'out-label1:out-label2']
        for i in range(len(inp_parameters)):
            inp_parameters[i] = re.split('\[|\]', inp_parameters[i])

        # command[0] - 'in-label1:in-label2...'
        # command[1] - 'filter_name:param1:param2...'
        # command[2] - 'out-label1:out-label2...'
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

            # now for every filter we specify dependencies
            inp_labels_splited = command[2].split(':')
            for ind in range(len(inp_labels_splited)):
                # mark the label as an output file if it is marked with -o=
                if inp_labels_splited[ind][0:3] == '-o=':
                    inp_labels_splited[ind] = inp_labels_splited[ind][3::]
                    res_obj.label_in_map[inp_labels_splited[ind]].calls_counter += 1
                    res_obj.fin_labels.append(inp_labels_splited[ind])  # update the list of final labels

                # create dependencies and update calls_counters
                res_obj.label_dependencies[inp_labels_splited[ind]] = []
                for in_label in command[0].split(':'):
                    if in_label not in res_obj.label_in_map and in_label[0:3] != '-i=':
                        raise WrongDependencyException(command[1][0], in_label)
                    res_obj.label_dependencies[inp_labels_splited[ind]].append(in_label)
                    if in_label[0:3] != '-i=':  # don't increase calls_counter if the label is an input image
                        res_obj.label_in_map[in_label].calls_counter += 1
                # store all list of out labels to restore indices later
                # (to choose input label, corresponding to the index of the out-label)
                res_obj.labels_to_out[inp_labels_splited[ind]] = inp_labels_splited

        # trace for debug
        # print("\nDependencies:")
        # for key in res_obj.label_dependencies:
        #     print(key, res_obj.label_dependencies[key])
        #
        # print("\nLabels map to filters:")
        # for key in res_obj.label_in_map:
        #     print(key, res_obj.label_in_map[key], "calls:", res_obj.label_in_map[key].calls_counter)
        #
        # print("\nLabels to out:")
        # for key in res_obj.labels_to_out:
        #     print(key, res_obj.labels_to_out[key])

        # trace for user
        print("\nFiles with the following names will be created:")
        for label in res_obj.fin_labels:
            print(label)

        # check if input files are incorrect
        for key in res_obj.label_dependencies:
            if res_obj.label_dependencies[key][0][0:3] == '-i=' and \
                    not os.path.exists(prefix + '/' + res_obj.label_dependencies[key][0][3::]):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        prefix + '/' +res_obj.label_dependencies[key][0][3::])

        print()
        return res_obj
