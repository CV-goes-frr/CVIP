# CVIP

CVIP stands for Computer Vision Image Processing tool, that is a convenient, fast and flexible command line tool for editing pictures with the use of Computer Vision methods.

Main features:
--------------
  - `Blurring faces` (photo/video)
  - `Crop` (photo/video)
  - `Overlaying masking` (photo/video)
  - `Face detection` (photo/video)
  - `Upscaling/Downscaling with some scale factor` (photo/video)
  - `Upscaling/Downscaling to some resolution` (photo/video)
  - `Motion Tracking` (video)
  - `Saturation` (photo/video)
  - `Flip horizontally/vertically` (photo/video)
  - `Overlay one video with another` (video)
  - `Reverse` (video)
  - `Panorama` (video)
  - `Feature Matching` (photo/video)

## Building:
Open the root directory of the CVIP project you've downloaded. There should be CVIP.spec file.
Building should be done with pyinstaller with the following command:

`pyinstaller CVIP.spec`

After that, pyinstaller will create `build` directory and `dist` directory.
Executable file of CVIP will be placed in the `dist` directory.
