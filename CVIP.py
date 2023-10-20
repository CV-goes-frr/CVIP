import cv2

from src.Parser import Parser

def main():
    while True:
        user_input = input("Enter a prompt or type 'exit' to quit or 'help' for information: ")

        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'help':
            with open('help.txt', 'r') as help_file:
                print(help_file.read())
            continue

        prompt = user_input
        processes_limit = input("Enter the number of parallel processes: ")

        pars = Parser(prompt, processes_limit)
        proc = pars.parse()

        print("\nPROCESSING...\n")
        for fin in proc.fin_labels:
            res_images_list = proc.process(fin)
            for res_img_index in range(len(res_images_list)):
                cv2.imwrite(f'{fin}{res_img_index}.jpg', res_images_list[res_img_index])

if __name__ == "__main__":
    main()
