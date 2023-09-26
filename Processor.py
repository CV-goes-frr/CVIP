import sys


class Filter:

    def __init__(self):
        self.filter_in = []
        self.img_in = ""


class Processor:

    def __init__(self, result_label: Filter):
        self.head = result_label
        self.map = {"crop": Crop}

class Parser:




def main():
    filename = sys.argv[0]
  


if __name__ == "__main__":
    main()
