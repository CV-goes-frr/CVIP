# [![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) 
# CVIP
CVIP stands for Computer Vision Image Processing tool, that is an open-source, convenient, fast and flexible command line tool for editing pictures with the use of Computer Vision methods.

Main features:
--------------
# `Blurring faces` (photo/video)
## How to use?
`./CVIP [-i=../examples/man.jpg]face_blur:10[-o=../examples/blurred_man]` with photos

![Original](examples/man.jpg "Original") 
![Result](examples/blurred_man.jpg "Result")
![Original](examples/group.jpg "Original") 
![Result](examples/blurred_group.jpg "Result")

`./CVIP -v [-i=your_video.mp4]face_blur:10[-o=out]` with videos
![Original](examples/woman.gif "Original") 
![Result](examples/blurred_woman.gif "Result")

# `Crop` (photo/video)
## How to use?
`./CVIP [-i=../examples/man.jpg]crop:10:35:320:300[-o=../examples/cropped_man]` with photos

![Result](examples/cropped_man.jpg "Result")

`./CVIP -v [-i=../examples/woman.mov]crop:100:350:520:500[-o=../examples/cropped_woman]` with videos

![Result](examples/cropped_woman.gif "Result")

# `Overlaying masking` (photo/video)
## How to use?
`./CVIP [-i=../examples/man.jpg]mask:../examples/mask.jpg[-o=../examples/man_with_a_mask]` with photos

![Result](examples/man_with_a_mask.jpg "Result")
![Result](examples/group_with_a_mask.jpg "Result")

`./CVIP -v [-i=../examples/woman.mov]mask:../examples/mask.jpg[-o=../examples/woman_with_a_mask]` with videos

![Result](examples/woman_with_a_mask.gif "Result")

# `Face detection` (photo/video)
## How to use?
`./CVIP [-i=../examples/man.jpg]face_detection[-o=../examples/detected_face_man]` with photos

![Result](examples/detected_face_man.jpg "Result")
![Result](examples/detected_face_group.jpg "Result")

`./CVIP -v [-i=../examples/woman.mov]face_detection[-o=../examples/detected_face_woman]` with videos

![Result](examples/detected_face_woman.gif "Result")

# `Upscaling/Downscaling with some scale factor` (photo/video)
## How to use?
`./CVIP [-i=../examples/man.jpg]nn_scale_with_factor:2[-o=../examples/man_x2_nn]` with photos
`./CVIP [-i=../examples/man.jpg]bilinear_scale_with_factor:2[-o=../examples/man_x2_bilinear]` with photos

![Result](examples/scale.jpg "Result")

`./CVIP -v [-i=../examples/woman.mov]nn_scale_with_factor:2[-o=../examples/woman_x2_nn]` with videos
`./CVIP -v [-i=../examples/woman.mov]bilinear_scale_with_factor:2[-o=../examples/woman_x2_bilinear]` with videos


# `Upscaling/Downscaling to some resolution` (photo/video)
## How to use?
`./CVIP [-i=../examples/man.jpg]scale_to_resolution:1000:1000[-o=../examples/man_1000_1000]` with photos
`./CVIP -v [-i=../examples/woman.mov]scale_to_resolution:1000:1000[-o=../examples/woman_1000_1000]` with videos

![Result](examples/patrick.jpg "Result")

# `Motion Tracking` (video)
## How to use?
`./CVIP -v [-i=../examples/run.mov]motion_tracking[-o=../examples/run_motion]` with videos

![Result](examples/motion_tracking_example.gif "Result")

# `Saturation` (photo)
## How to use?
`./CVIP [-i=../examples/sunset.jpg]saturation:2[-o=../examples/saturation]` with photos

![Result](examples/sunset.png "Result")

# `Flip horizontally/vertically` (photo/video)
## How to use?
`./CVIP [-i=../examples/man.jpg]flip:horizontal[-o=../examples/man_flipped_hor]` with photos

![Result](examples/man_flipped_hor.jpg "Result")

`./CVIP -v [-i=../examples/woman.mov]flip:vertical[-o=../examples/woman_flipped_vert]` with videos

![Result](examples/woman_flipped_vert.gif "Result")

# `Overlay one video with another` (video)
## How to use?

`./CVIP -v [-i=../examples/RIHANNA_DIAMONDS.mp4]video_overlay:../examples/surdo.mp4:4:10:30:1[-o=../examples/rihanna_overlayed]` with videos

![Result](examples/rihanna_overlayed.gif "Result")

# `Reverse` (video)
## How to use?
`./CVIP [-i=../examples/woman.mov]reverse[-o=../examples/woman_reversed]` with videos

![Result](examples/woman_reversed.gif "Result")

# `Panorama` (video)
## How to use?


# `Feature Matching` (photo/video)
## How to use?
`./CVIP [-i=match1.png]feature_matching:BF:match2.png[-o=matched]`

![Result](examples/object_detect.jpg "Result")


# Building:
Open the root directory of the CVIP project you've downloaded. There should be CVIP.spec file.
Building should be done with pyinstaller with the following command:

`pyinstaller CVIP.spec`

After that, pyinstaller will create `build` directory and `dist` directory.
Executable file of CVIP will be placed in the `dist` directory.
