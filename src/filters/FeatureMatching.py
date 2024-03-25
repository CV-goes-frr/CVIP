import os
import sys
from multiprocessing import Pool
from typing import List
from settings import prefix

import numpy as np
import cv2
from .Filter import Filter


class FeatureMatching(Filter):

    def __init__(self, type_match: str, img2: str):
        super().__init__()  # Call the constructor of the parent class (Filter)
        self.type_match = type_match
        self.img2 = cv2.imread(f'{prefix}/{img2}')


    def apply(self, img1: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:

        print("FEATURE MATCHING IN PROGRESS...")
        if self.cache:  # Check if a cached result exists
            print("USING CACHE...")
            return self.cache  # Return the cached result

        img_copy1 = np.copy(img1)
        img_copy2 = np.copy(self.img2)
        gray1 = cv2.GaussianBlur(img_copy1, (5, 5), 0)
        gray2 = cv2.GaussianBlur(img_copy2, (5, 5), 0)

        gray1 = cv2.cvtColor(gray1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(gray2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

        if self.type_match == 'BF':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(descriptors1, descriptors2)

            # Sort matches based on their distances
            matches = sorted(matches, key=lambda x: x.distance)

            # Set a distance threshold
            distance_threshold = 100
            good_matches = [match for match in matches if match.distance < distance_threshold]
        elif self.type_match == 'FLANN':
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < 0.9 * n.distance:
                        good_matches.append(m)

        minMatches = 10
        if len(good_matches) > minMatches:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w, _ = img_copy1.shape
            object_bbox = cv2.boundingRect(np.intp(src_pts))

            pts = np.float32([[[object_bbox[0], object_bbox[1]],
                               [object_bbox[0], object_bbox[1] + object_bbox[3]],
                               [object_bbox[0] + object_bbox[2],
                                object_bbox[1] + object_bbox[3]],
                               [object_bbox[0] + object_bbox[2], object_bbox[1]]]]).reshape(-1, 1, 2)
            dst = np.int32(cv2.perspectiveTransform(pts, M))
            image2 = cv2.polylines(img_copy2, [dst], True, 255, 3, cv2.LINE_AA)
        else:
            print("Not enough features")
        # Draw only the good matches
        img_copy = cv2.drawMatches(img_copy1, keypoints1, img_copy2, keypoints2, good_matches, None,
                                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        if self.calls_counter > 1:  # Check if the method has been called more than once
            self.cache = [img_copy]  # Cache the upscaled image
        return [img_copy]  # Return the edited image as a list

