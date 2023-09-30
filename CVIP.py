import sys
import os
import errno

import numpy as np
import cv2
import argparse

from Maintenance.Parser import Parser


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
    
    # create out file for every result image (there can be more than one)
    for i in range(len(img)):
        cv2.imwrite(f'out{i}.jpg', img[i])


if __name__ == "__main__":
    main()