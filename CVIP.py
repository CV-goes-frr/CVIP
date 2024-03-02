import time
import argparse

import cv2

from settings import prefix
from src.VerifyQuery import VerifyQuery
from src.exceptions.InvalidBrackets import InvalidBracketsException
from src.exceptions.InvalidFlags import InvalidFlagsException
from src.exceptions.WrongFileFormat import WrongFileFormatException
from src.exceptions.WrongFilename import WrongFilenameException
from src.exceptions.WrongFiltername import WrongFilterNameException
from src.exceptions.WrongDependency import WrongDependencyException
from src.exceptions.WrongParameters import WrongParametersException
from src.Parser import Parser
from src.exceptions.NoFace import NoFaceException


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-h",
        "--help",
        action="store_true")
    parser.add_argument(
        "-v",
        "--video",
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
    elif args.video:
        if args.actions is None:
            print("Give CVIP a prompt or check help with -h or --help...")
            return
        try:
            process_lim = 1
            if args.parallel_processes:
                process_lim = args.parallel_processes

            # Checking prompt correctness
            # VerifyQuery.check(args.actions)  # if something is wrong exceptions will occur

            pars = Parser(args.actions, process_lim)
            proc = pars.parse(video_editing=True)

            # writer
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

            start: float = time.time()
            print("\nPROCESSING...\n")
            for fin in proc.fin_labels:
                res_images_list = proc.process(fin)
                out = cv2.VideoWriter(f'{prefix}/{fin}.mp4', fourcc, proc.fps, (proc.width, proc.height))
                # write result frame by frame
                for value in res_images_list[0]:
                    out.write(value)

                out.release()  # Обязательно нужно закрыть

                # create as many out files for every final label as we want
                # for res_img_index in range(len(res_images_list)):
                #     cv2.imwrite(f'{prefix}/{fin}{res_img_index}.jpg', res_images_list[res_img_index])
            end: float = time.time()
            print(f"\nALL TASKS WERE COMPLETED\nTIME ELAPSED: {end - start}\n")

        except (WrongDependencyException, WrongFilterNameException, WrongParametersException,
                NoFaceException, InvalidBracketsException, InvalidFlagsException, WrongFileFormatException,
                WrongFilenameException) as e:
            print("\n\n!!! Error occurred !!!\n" + str(e) + "\n")
        except FileNotFoundError as e:
            print("\n\n!!! Error occurred !!!\n" + str(e) + "\n")
    else:
        if args.actions is None:
            print("Give CVIP a prompt or check help with -h or --help...")
            return
        try:
            process_lim = 1
            if args.parallel_processes:
                process_lim = args.parallel_processes

            # Checking prompt correctness
            VerifyQuery.check(args.actions)  # if something is wrong exceptions will occur

            pars = Parser(args.actions, process_lim)
            proc = pars.parse(video_editing=False)

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
        except (WrongDependencyException, WrongFilterNameException, WrongParametersException,
                NoFaceException, InvalidBracketsException, InvalidFlagsException, WrongFileFormatException,
                WrongFilenameException) as e:
            print("\n\n!!! Error occurred !!!\n" + str(e) + "\n")
        except FileNotFoundError as e:
            print("\n\n!!! Error occurred !!!\n" + str(e) + "\n")

        
if __name__ == "__main__":
    main()
