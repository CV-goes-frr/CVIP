import numpy as np
import math
import os
import errno
import cv2
import argparse


class Filter:

    def __init__(self,img):
        self.input_image = cv2.imread(img) # open image
        self.input_height, self.input_width, _ = self.input_image.shape # initialize sizes


class Crop(Filter):
    
    def apply(self, x1, y1, x2, y2):
        cropped_image = self.input_image[y1:y2, x1:x2]
        return cropped_image


class NnScale(Filter):

    def apply(self, scale_factor):
        new_width = int(self.input_width * scale_factor)
        new_height = int(self.input_height * scale_factor)

        upscaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        for y in range(new_height):
            for x in range(new_width):
                original_x = int(x / scale_factor)
                original_y = int(y / scale_factor)

                upscaled_image[y, x] = self.input_image[original_y, original_x]

        return upscaled_image


class BilinearScale(Filter):

    def apply(self, scale_factor):
        new_width = int(self.input_width * scale_factor)
        new_height = int(self.input_height * scale_factor)

        upscaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        for y in range(new_height):
            for x in range(new_width):
                original_x = int(x / scale_factor)
                original_y = int(y / scale_factor)

                x1, y1 = int(original_x), int(original_y)
                x2, y2 = x1 + 1, y1 + 1

                x1 = min(max(x1, 0), self.input_width - 1)
                x2 = min(max(x2, 0), self.input_width - 1)
                y1 = min(max(y1, 0), self.input_height - 1)
                y2 = min(max(y2, 0), self.input_height - 1)

                alpha = original_x - x1
                beta = original_y - y1

                top_left = self.input_image[y1, x1]
                top_right = self.input_image[y1, x2]
                bottom_left = self.input_image[y2, x1]
                bottom_right = self.input_image[y2, x2]

                weight = (1 - alpha) * (1 - beta) * top_left + alpha * (1 - beta) * top_right + (1 - alpha) * beta * bottom_left + alpha * beta * bottom_right

                upscaled_image[y, x] = weight.astype(np.uint8)

        return upscaled_image


class BicubicScale(Filter):

    def kernel_equation(self, coeff, x):
        if abs(x) <= 1:
            return (coeff + 2) * (abs(x)**3) - (coeff + 3) * (abs(x)**2) + 1
        elif abs(x) > 1 and abs(x) <= 2:
            return coeff * (abs(x)**3) - 5 * coeff * (abs(x)**2) + 8 * coeff * abs(x) - 4 * coeff
        else:    
            return 0

    def apply(self, scale_factor):
        coeff = -0.5

        new_width, new_height = math.floor(self.input_width * scale_factor), math.floor(self.input_height * scale_factor)
        color = 3

        matriximage=np.zeros((new_height, new_width, 3))

        for c in range(color):
            for w in range(new_width):
                for h in range(new_height):
                    x = w / scale_factor
                    y = h / scale_factor

                    x1 = min(max(math.floor(x) - 1, 0), self.input_width - 1)
                    x2 = min(max(math.floor(x), 0), self.input_width - 1)
                    x3 = min(max(math.floor(x) + 1, 0), self.input_width - 1)
                    x4 = min(max(math.floor(x) + 2, 0), self.input_width - 1)

                    y1 = min(max(math.floor(y) - 1, 0), self.input_height - 1)
                    y2 = min(max(math.floor(y), 0), self.input_height - 1)
                    y3 = min(max(math.floor(y) + 1, 0), self.input_height - 1)
                    y4 = min(max(math.floor(y) + 2, 0), self.input_height - 1)
                
                    mat_kernelx = np.matrix([[self.kernel_equation(x - x1, coeff),
                                              self.kernel_equation(x - x2, coeff),
                                              self.kernel_equation(x - x3, coeff),
                                              self.kernel_equation(x - x4, coeff)]])
                    
                    mat_near = np.matrix([[self.input_image[int(y1), int(x1), c],
                                           self.input_image[int(y2), int(x1), c],
                                           self.input_image[int(y3), int(x1), c],
                                           self.input_image[int(y4), int(x1), c]],
                                          [self.input_image[int(y1), int(x2), c],
                                           self.input_image[int(y2), int(x2), c],
                                           self.input_image[int(y3), int(x2), c],
                                           self.input_image[int(y4), int(x2), c]],
                                          [self.input_image[int(y1), int(x3), c],
                                           self.input_image[int(y2), int(x3), c],
                                           self.input_image[int(y3), int(x3), c],
                                           self.input_image[int(y4), int(x3), c]],
                                          [self.input_image[int(y1), int(x4), c],
                                           self.input_image[int(y2), int(x4), c],
                                           self.input_image[int(y3), int(x4), c],
                                           self.input_image[int(y4), int(x4), c]]])
                    
                
                    mat_kernely = np.matrix([[self.kernel_equation(y - y1, coeff)],
                                             [self.kernel_equation(y - y2, coeff)],
                                             [self.kernel_equation(y - y3, coeff)],
                                             [self.kernel_equation(y - y4, coeff)]])

                    matriximage[h, w, c] = np.dot(np.dot(mat_kernelx, mat_near), mat_kernely)

                    if matriximage[h, w, c] > 255:
                        matriximage[h, w, c] = 255
                    elif matriximage[h, w, c] < 0:
                        matriximage[h, w, c] = 0

        return matriximage


class Tools:

    def __init__(self,img):
        self.input_image = cv2.imread(img) # open image
        self.input_height, self.input_width, _ = self.input_image.shape # initialize sizes
        '''self.pow2_cache = {} #cache for powers
        self.pow3_cache = {} #cache for powers'''
  

    def crop(self, x1, y1, x2, y2):
        cropped_image = self.input_image[y1:y2, x1:x2]
        return cropped_image


    def bilinear_scale(self, scale_factor):
        new_width = int(self.input_width * scale_factor)
        new_height = int(self.input_height * scale_factor)

        upscaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        for y in range(new_height):
            for x in range(new_width):
                original_x = int(x / scale_factor)
                original_y = int(y / scale_factor)

                x1, y1 = int(original_x), int(original_y)
                x2, y2 = x1 + 1, y1 + 1

                x1 = min(max(x1, 0), self.input_width - 1)
                x2 = min(max(x2, 0), self.input_width - 1)
                y1 = min(max(y1, 0), self.input_height - 1)
                y2 = min(max(y2, 0), self.input_height - 1)

                alpha = original_x - x1
                beta = original_y - y1

                top_left = self.input_image[y1, x1]
                top_right = self.input_image[y1, x2]
                bottom_left = self.input_image[y2, x1]
                bottom_right = self.input_image[y2, x2]

                weight = (1 - alpha) * (1 - beta) * top_left + alpha * (1 - beta) * top_right + (1 - alpha) * beta * bottom_left + alpha * beta * bottom_right

                upscaled_image[y, x] = weight.astype(np.uint8)

        return upscaled_image


    def nn_scale(self, scale_factor):
        new_width = int(self.input_width * scale_factor)
        new_height = int(self.input_height * scale_factor)

        upscaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        for y in range(new_height):
            for x in range(new_width):
                original_x = int(x / scale_factor)
                original_y = int(y / scale_factor)

                upscaled_image[y, x] = self.input_image[original_y, original_x]

        return upscaled_image


    def kernel_equation(self, coeff, x):
        if abs(x) <= 1:
            return (coeff + 2) * (abs(x)**3) - (coeff + 3) * (abs(x)**2) + 1
        elif abs(x) > 1 and abs(x) <= 2:
            return coeff * (abs(x)**3) - 5 * coeff * (abs(x)**2) + 8 * coeff * abs(x) - 4 * coeff
        else:    
            return 0


    def bicubic_scale(self, scale_factor):
        coeff = -0.5

        new_width, new_height = math.floor(self.input_width * scale_factor), math.floor(self.input_height * scale_factor)
        color = 3

        matriximage=np.zeros((new_height, new_width, 3))

        for c in range(color):
            for w in range(new_width):
                for h in range(new_height):
                    x = w / scale_factor
                    y = h / scale_factor

                    x1 = min(max(math.floor(x) - 1, 0), self.input_width - 1)
                    x2 = min(max(math.floor(x), 0), self.input_width - 1)
                    x3 = min(max(math.floor(x) + 1, 0), self.input_width - 1)
                    x4 = min(max(math.floor(x) + 2, 0), self.input_width - 1)

                    y1 = min(max(math.floor(y) - 1, 0), self.input_height - 1)
                    y2 = min(max(math.floor(y), 0), self.input_height - 1)
                    y3 = min(max(math.floor(y) + 1, 0), self.input_height - 1)
                    y4 = min(max(math.floor(y) + 2, 0), self.input_height - 1)
                
                    mat_kernelx = np.matrix([[self.kernel_equation(x - x1, coeff),
                                              self.kernel_equation(x - x2, coeff),
                                              self.kernel_equation(x - x3, coeff),
                                              self.kernel_equation(x - x4, coeff)]])
                    
                    mat_near = np.matrix([[self.input_image[int(y1), int(x1), c],
                                           self.input_image[int(y2), int(x1), c],
                                           self.input_image[int(y3), int(x1), c],
                                           self.input_image[int(y4), int(x1), c]],
                                          [self.input_image[int(y1), int(x2), c],
                                           self.input_image[int(y2), int(x2), c],
                                           self.input_image[int(y3), int(x2), c],
                                           self.input_image[int(y4), int(x2), c]],
                                          [self.input_image[int(y1), int(x3), c],
                                           self.input_image[int(y2), int(x3), c],
                                           self.input_image[int(y3), int(x3), c],
                                           self.input_image[int(y4), int(x3), c]],
                                          [self.input_image[int(y1), int(x4), c],
                                           self.input_image[int(y2), int(x4), c],
                                           self.input_image[int(y3), int(x4), c],
                                           self.input_image[int(y4), int(x4), c]]])
                    
                
                    mat_kernely = np.matrix([[self.kernel_equation(y - y1, coeff)],
                                             [self.kernel_equation(y - y2, coeff)],
                                             [self.kernel_equation(y - y3, coeff)],
                                             [self.kernel_equation(y - y4, coeff)]])

                    matriximage[h, w, c] = np.dot(np.dot(mat_kernelx, mat_near), mat_kernely)

                    if matriximage[h, w, c] > 255:
                        matriximage[h, w, c] = 255
                    elif matriximage[h, w, c] < 0:
                        matriximage[h, w, c] = 0

        return matriximage


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type = str, help = 'Path to input image')
    parser.add_argument('--crop', nargs = 4, type = int, metavar = ('x1', 'y1', 'x2', 'y2'), help = 'Crop the image')
    parser.add_argument('--nn_scale', type = float, help = 'Scaling the image with nearest neighbours method')
    parser.add_argument('--bilinear_scale', type = float, help = 'Scaling the image with bilinear method')
    parser.add_argument('--bicubic_scale', type = float, help = 'Scaling the image with bicubic method')

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
      raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.image_path)

    image = Tools(args.image_path)

    if args.crop:
        x1, y1, x2, y2 = args.crop
        cropped_image = image.crop(x1, y1, x2, y2)
        cv2.imwrite('cropped_image.jpg', cropped_image)
        
    if args.nn_scale:
        scale_factor = args.nn_scale
        resized_image = image.nn_scale(scale_factor)
        cv2.imwrite('resized_image.jpg', resized_image)
    
    if args.bilinear_scale:
        scale_factor = args.bilinear_scale
        resized_image = image.bilinear_scale(scale_factor)
        cv2.imwrite('resized_image.jpg', resized_image)

    if args.bicubic_scale:
        scale_factor = args.bicubic_scale
        resized_image = image.bicubic_scale(scale_factor)
        cv2.imwrite('resized_image.jpg', resized_image)


if __name__ == "__main__":
    main()
