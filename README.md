# [![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) 
# CVIP
CVIP stands for Computer Vision Image Processing tool, that is a convenient, fast and flexible command line tool for editing pictures with the use of Computer Vision methods.

Main features:
--------------
  - `Blurring faces` (photo/video)
![blur](resources/gentlemen.jpg)
./CVIP [-i=Gentlemen.png]face_blur:10[-o=blurred]
  - `Crop` (photo/video)
![crop](resources/bad.png)
./CVIP [-i=walter.png]crop:700:650:850:730[-o=eye]
  - `Overlaying masking` (photo/video)
![mask](resources/masking.png)
./CVIP [-i=photo.jpg]mask:man.jpg[-o=masked]
  - `Face detection` (photo/video)
![face](resources/faces.jpg)
./CVIP [-i=photo.jpg]face_detection[-o=detected]
  - `Upscaling/Downscaling with some scale factor` (photo/video)
![scale](resources/scale.jpg)
./CVIP [-i=animals.png]nn_scale_with_factor:2[-o=scaled1]
./CVIP [-i=animals.png]bilinear_scale_with_factor:2[-o=scaled2]
  - `Upscaling/Downscaling to some resolution` (photo/video)
![resolution](resources/patrick.jpg)
./CVIP [-i=bateman.png]scale_to_resolution:1280:300[-o=blooper]
  - `Motion Tracking` (video)
![motion](resources/motion_tracking_example.gif)
./CVIP -v [-i=runner.mp4]motion_tracking[-o=tracked]
  - `Saturation` (photo/video)
![saturation](resources/sunset.png)
./CVIP [-i=sunset.png]saturation:1.2[-o=saturated]
  - `Flip horizontally/vertically` (photo/video)
  - `Overlay one video with another` (video)
  - `Reverse` (video)
  - `Panorama` (video)
  - `Feature Matching` (photo/video)
![matching](resources/object_detect.jpg)
./CVIP [-i=match1.png]feature_matching:BF:match2.png[-o=matched_showcase]

## Building:
Open the root directory of the CVIP project you've downloaded. There should be CVIP.spec file.
Building should be done with pyinstaller with the following command:

`pyinstaller CVIP.spec`

After that, pyinstaller will create `build` directory and `dist` directory.
Executable file of CVIP will be placed in the `dist` directory.
