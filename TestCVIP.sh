./CVIP [-i=face.jpg]scale_to_resolution:1000:1000[-o=scaled_to_factor_parallel] --parallel_processes=4
./CVIP [-i=photo.jpg]crop:0:0:100:100[-o=cropped]
./CVIP [-i=face.jpg]nn_scale_with_factor:2[-o=nn_scaled] --parallel_processes=4
./CVIP [-i=face.jpg]bilinear_scale_with_factor:2[-o=bilinear_scaled] --parallel_processes=4
./CVIP [-i=photo.jpg]face_detection[-o=detected]
./CVIP [-i=photo.jpg]face_blur:3[-o=blurred]
./CVIP [-i=photo.jpg]mask:face.jpg[-o=masked]

