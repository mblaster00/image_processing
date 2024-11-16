import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton


class GenNearest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """

    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, skeleton):
        """Generates an image from the given skeleton."""
        min_distance = float('inf')
        best_frame_idx = None

        # Iterate over each frame's skeleton in the target video
        for frame_idx, video_skeleton in enumerate(self.videoSkeletonTarget.ske):
            # Compute the distance between the given skeleton and the target skeleton
            try:
                distance = skeleton.distance(video_skeleton)
            except ValueError as e:
                print(f"Distance computation failed for frame {frame_idx}: {e}")
                continue

            # Update the minimum distance and best frame index
            if distance < min_distance:
                min_distance = distance
                best_frame_idx = frame_idx

        # Retrieve and return the best frame's image
        if best_frame_idx is not None:
            best_frame_image = self.videoSkeletonTarget.readImage(best_frame_idx)
            return best_frame_image
        else:
            print("Error: No matching frame found.")
            return np.zeros((128, 128, 3), dtype=np.uint8)  # Error fallback



