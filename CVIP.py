import time
import argparse

import cv2

from settings import BASE_DIR
from src.Parser import Parser


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-h",
        "--help",
        action="store_true")
    parser.add_argument(
        'actions',
        type=str,
        nargs=argparse.OPTIONAL,
        help='Prompt')
    parser.add_argument(
        'processes_limit',
        type=str,
        nargs=argparse.OPTIONAL,
        help='Max number of parallel processes')
    args = parser.parse_args()

    if args.help:
        with open(BASE_DIR + '/help.txt', 'r') as help_file:
            print(help_file.read())
    else:
        pars = Parser(args.actions, args.processes_limit)
        proc = pars.parse()

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
