from sys import platform

if platform == "linux" or platform == "linux2":
    prefix = ''
elif platform == "darwin":
    prefix = '../'