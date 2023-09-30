import cv2
import argparse
import numpy as np
from typing import List
from Maintenance.Parser import Parser
from Maintenance.Processor import Processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('actions', type=str, help='Path to input image')
    args = parser.parse_args()

    pars: Parser = Parser(args.image_path, args.actions)
    proc: Processor = pars.parse()
    img: List[np.ndarray] = proc.process(proc.fin)

    # create out file for every result image (there can be more than one)
    for i in range(len(img)):
        cv2.imwrite(f'out{i}.jpg', img[i])


if __name__ == "__main__":
    main()
