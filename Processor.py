import sys
import os
import errno

import cv2
import numpy as np
import re
import argparse


class Filter:

    def __init__(self):
        self.filter_in = []
        self.img_in = ""


class Processor:

    def __init__(self):
        self.fin = ""
        crop = Crop
        nn = NnScale
        biln = BilinearScale
        self.class_map = {"crop": crop,
                          "nn_scale": nn,
                          "bilinear_scale": biln}

        # what in-labels should be already done for applying our filter with this out-label
        self.label_dependencies = {}

        # what filter is mapped for the label
        self.label_in_map = {}

        self.inp_image = ""
    
    def process(self, label):
        if label != "-i":
            if (label not in self.label_dependencies):
                raise Exception("Label doesn't exist: " + label)
            prev_label = self.label_dependencies[label]
            image = self.process(prev_label)
        else:
            image = cv2.imread(self.inp_image)
        
        if label != self.fin:
            return self.label_in_map[label].apply(image)
        else:
            return image


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
                    res_obj.label_dependencies[command[2]] = command[0]
                    res_obj.label_in_map[command[0]] = res_obj.class_map["crop"](command[1][1],
                                                                                 command[1][2],
                                                                                 command[1][3],
                                                                                 command[1][4])
                case 'nn_scale':
                    if len(command[1]) != 2:
                        raise Exception("Wrong number of parameters for nn_scale")
                    res_obj.label_dependencies[command[2]] = command[0]
                    res_obj.label_in_map[command[0]] = res_obj.class_map["nn_scale"](command[1][1])
                case 'bilinear_scale':
                    if len(command[1]) != 2:
                        raise Exception("Wrong number of parameters for nn_scale")
                    res_obj.label_dependencies[command[2]] = command[0]
                    res_obj.label_in_map[command[0]] = res_obj.class_map["bilinear_scale"](command[1][1])
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


class Crop():

    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = int(x1), int(y1), int(x2), int(y2)
    
    def apply(self, img):
        input_height, input_width, _ = img.shape
        if (self.x1 > input_width or self.y1 > input_height or self.x2 > input_width or self.y2 > input_height or
            (self.x1 >= self.x2) or (self.y1 >= self.y2) or self.x1 < 0 or self.x2 < 0 or self.y1 < 0 or self.y2 < 0 or
            type(self.x1) != int or type(self.x2) != int or type(self.y1) != int or type(self.y2) != int):
            raise Exception("Wrong crop parameters: " + self.x1 + ' ' + self.y1 + ' ' + self.x2 + ' ' + self.y2)
            
        cropped_image = img[self.y1:self.y2, self.x1:self.x2]
        return cropped_image


class NnScale():

    def __init__(self, scale_factor):
        self.scale_factor = float(scale_factor)

    def apply(self, img):
        input_height, input_width, _ = img.shape
        new_width = int(input_width * self.scale_factor)
        new_height = int(input_height * self.scale_factor)
        upscaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        for y in range(new_height):
            for x in range(new_width):
                original_x = int(x / self.scale_factor)
                original_y = int(y / self.scale_factor)

                upscaled_image[y, x] = img[original_y, original_x]

        return upscaled_image


class BilinearScale():

    def __init__(self, scale_factor):
        self.scale_factor = float(scale_factor)
    
    def apply(self, img):
        input_height, input_width, _ = img.shape
        new_width = int(input_width * self.scale_factor)
        new_height = int(input_height * self.scale_factor)

        upscaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        for y in range(new_height):
            for x in range(new_width):
                original_x = int(x / self.scale_factor)
                original_y = int(y / self.scale_factor)

                x1, y1 = int(original_x), int(original_y)
                x2, y2 = x1 + 1, y1 + 1

                x1 = min(max(x1, 0), input_width - 1)
                x2 = min(max(x2, 0), input_width - 1)
                y1 = min(max(y1, 0), input_height - 1)
                y2 = min(max(y2, 0), input_height - 1)

                alpha = original_x - x1
                beta = original_y - y1

                top_left = img[y1, x1]
                top_right = img[y1, x2]
                bottom_left = img[y2, x1]
                bottom_right = img[y2, x2]

                weight = (1 - alpha) * (1 - beta) * top_left + alpha * (1 - beta) * top_right + (1 - alpha) * beta * bottom_left + alpha * beta * bottom_right

                upscaled_image[y, x] = weight.astype(np.uint8)

        return upscaled_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type = str, help = 'Path to input image')
    parser.add_argument('actions', type = str, help = 'Path to input image')
    args = parser.parse_args()

    # print(args.image_path)
    # print(args.actions)
    pars = Parser(args.image_path, args.actions)
    proc = pars.parse()

    img = proc.process(proc.fin)
    cv2.imwrite('out.jpg', img)


if __name__ == "__main__":
    main()
