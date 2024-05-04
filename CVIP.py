import argparse
import os

import cv2
import time
from multiprocessing import freeze_support
from moviepy.editor import VideoFileClip

from settings import prefix
from src.Parser import Parser
from src.VerifyQuery import VerifyQuery
from src.exceptions.InvalidBrackets import InvalidBracketsException
from src.exceptions.InvalidFlags import InvalidFlagsException
from src.exceptions.NoFace import NoFaceException
from src.exceptions.WrongDependency import WrongDependencyException
from src.exceptions.WrongFileFormat import WrongFileFormatException
from src.exceptions.WrongFilename import WrongFilenameException
from src.exceptions.WrongFiltername import WrongFilterNameException
from src.exceptions.WrongParameters import WrongParametersException
from src.filters.VideoToPanorama import VideoToPanorama


def main():
    """
    Main function to execute the program.
    """
    freeze_support()
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
            VerifyQuery.check(args.actions)  # if something is wrong exceptions will occur

            pars = Parser(args.actions, process_lim)
            proc = pars.parse(video_editing=True)

            # writer
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

            start: float = time.time()
            print("\nPROCESSING...\n")
            for fin in proc.fin_labels:
                res_images_list = proc.process(fin)
                out = cv2.VideoWriter(f'{prefix}/{fin}_.mp4', fourcc, proc.fps, (proc.width, proc.height))
                # write result frame by frame
                for value in res_images_list[0]:
                    out.write(value)

                out.release()  # It's necessary

                # merge the audio
                video = VideoFileClip(f'{prefix}/{fin}_.mp4')
                video.without_audio()
                video_merged = video.set_audio(proc.audio)
                video_merged.write_videofile(f'{prefix}/{fin}.mp4')
                os.remove(f'{prefix}/{fin}_.mp4')

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
            if sum(isinstance(x, VideoToPanorama) for x in proc.label_in_map.values()):
                proc.video_editing = True

            start: float = time.time()
            print("\nPROCESSING...\n")
            for fin in proc.fin_labels:
                res_images_list = proc.process(fin)
                cv2.imwrite(f'{prefix}/{fin}.jpg', res_images_list[0])

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
