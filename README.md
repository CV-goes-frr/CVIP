# [![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit) CVIP
CVIP stands for Computer Vision Image Processing tool, that is a convenient, fast and flexible command line tool for editing pictures with the use of Computer Vision methods.

Main features:
--------------
  - Blurring faces
  - Overlaying masking
  - Face detection
  - Upscaling/Downscaling with some scale factor
  - Upscaling/Downscaling to some resolution

## Building:
Open the root directory of the CVIP project you've downloaded. There should be CVIP.spec file.
Building should be done with pyinstaller with the following command:

`pyinstaller CVIP.spec`

After that, pyinstaller will create `build` directory and `dist` directory.
Executable file of CVIP will be placed in the `dist` directory.
