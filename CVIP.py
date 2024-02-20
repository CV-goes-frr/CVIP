import time
import argparse

import cv2

from settings import prefix
from src.exceptions import WrongDependency, WrongFiltername, WrongParameters
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
        '--parallel_processes',
        type=int,
        help='Max number of parallel processes')
    args = parser.parse_args()

    if args.help:
        with open('help.txt', 'r') as help_file:
            print(help_file.read())
    else:
        try:
            process_lim = 1
            if args.parallel_processes:
                process_lim = args.parallel_processes
            pars = Parser(args.actions, process_lim)
            proc = pars.parse()

            start: float = time.time()
            print("\nPROCESSING...\n")
            for fin in proc.fin_labels:
                res_images_list = proc.process(fin)
                cv2.imwrite(f'{prefix}/{fin}.jpg', res_images_list[0])
                # create as many out files for every final label as we want
                # for res_img_index in range(len(res_images_list)):
                #     cv2.imwrite(f'{prefix}/{fin}{res_img_index}.jpg', res_images_list[res_img_index])
            end: float = time.time()
            print(f"\nALL TASKS WERE COMPLETED\nTIME ELAPSED: {end - start}\n")
        except WrongDependency as err:
            print("Unknown dependency:", err)
        except WrongFiltername as err:
            print("Call of the unknown operation:", err)
        except WrongParameters as err:
            print("Wrong set of parameters for the operation:", err)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

        
if __name__ == "__main__":
    main()
