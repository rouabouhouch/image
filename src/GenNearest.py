
import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton



class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):
        """
        Generate an image from the closest skeleton in the target video.
        Uses Skeleton.distance() to find nearest neighbor.
        """
        min_dist = float('inf')
        closest_idx = 0

        # Loop over all skeletons in the target VideoSkeleton
        for idx, tgt_ske in enumerate(self.videoSkeletonTarget.ske):
            dist = ske.distance(tgt_ske)
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx

        # Load corresponding image
        img = self.videoSkeletonTarget.readImage(closest_idx)
        if img is None:
            # fallback red image if reading fails
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            img[:, :] = (0, 0, 255)
        else:
            img = img.astype(np.float32) / 255.0  # normalize to [0,1]

        return img
