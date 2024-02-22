mkdir -p test_results
./CVIP [-i=face.jpg]scale_to_resolution:1000:1000[-o=test_results/scaled_to_factor_parallel] --parallel_processes=4
./CVIP [-i=photo.jpg]crop:0:0:100:100[-o=test_results/cropped]
./CVIP [-i=face.jpg]nn_scale_with_factor:2[-o=test_results/nn_scaled] --parallel_processes=4
./CVIP [-i=face.jpg]bilinear_scale_with_factor:2[-o=test_results/bilinear_scaled] --parallel_processes=4
./CVIP [-i=photo.jpg]face_blur:3[-o=test_results/blurred]
./CVIP [-i=photo.jpg]mask:face.jpg[-o=test_results/masked]
./CVIP [-i=photo 2.jpg]mask:face.jpg[-o=test_results/masked_with_spacename]

echo "-----------"
echo "Exceptions:"
./CVIP [-i=none.jpg]face_blur:3[none]
./CVIP [-i=photo.jpg]face_blur:a[none]
./CVIP [-i=photo.jpg]face_blur:2[ok][what]face_blur:5[out]
./CVIP [-i=photo.jpg]whaaaat:okaaaay[none]

echo "-----------------"
echo "Prompt exceptions"
./CVIP [-i=filename.png]filter[-o=out]filter
./CVIP [-i=filename.png]filter[-o=out]filter[-i=filename.png]filter[-o=out]
./CVIP [-a=filename.png]filter[-o=out][-i=filename.png]filter[-o=out]
./CVIP [-i=filename.png]\ filter[-o=out][-i=filename.png]filter[-o=out]
./CVIP [-i\ =filename.png]filter[-o=out][-i=filename.png]filter[-o=out]
./CVIP [-i=filename.png][]][-o=out][-i=filename.png]filter[-o=out]

