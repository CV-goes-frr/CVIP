import time

import cv2
import argparse

from src.Parser import Parser
from src.Processor import Processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('actions', type=str, help='Prompt')
    parser.add_argument('processes_limit', type=str, help='Max number of parallel processes')
    args = parser.parse_args()

    pars: Parser = Parser(args.actions, args.processes_limit)
    proc: Processor = pars.parse()

    start: float = time.time()
    print("\nPROCESSING...\n")
    for fin in proc.fin_labels:
        res_images_list = proc.process(fin)
        # create as many out files for every final label as we want
        for res_img_index in range(len(res_images_list)):
            cv2.imwrite(f'{fin}{res_img_index}.jpg', res_images_list[res_img_index])

    end: float = time.time()
    print(f"\nALL TASKS WERE COMPLETED\nTIME ELAPSED: {end - start}\n")


if __name__ == "__main__":
    main()
