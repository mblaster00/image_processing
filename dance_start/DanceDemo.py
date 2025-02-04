import numpy as np
import cv2
import os
import pickle
import sys

from VideoSkeleton import VideoSkeleton
from VideoSkeleton import combineTwoImages
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenNearest import GenNeirest
from GenVanillaNN import GenVanillaNN, GenVanillaNN2
from GenGAN import GenGAN


class DanceDemo:
    """ class that run a demo of the dance.
        The animation/posture from self.source is applied to character define self.target using self.gen
    """
    def __init__(self, filename_src, typeOfGen=2):
        self.target = VideoSkeleton("data/taichi1.mp4")
        self.source = VideoReader(filename_src)
        if typeOfGen == 1:           # Nearest
            print("Generator: GenNeirest")
            self.generator = GenNeirest(self.target)
        elif typeOfGen == 2:         # VanillaNN (Direct skeleton approach)
            print("Generator: GenSimpleNN (Direct)")
            self.generator = GenVanillaNN(self.target, loadFromFile=True)
        elif typeOfGen == 3:         # VanillaNN2 (Image-based approach)
            print("Generator: GenSimpleNN (Image-based)")
            self.generator = GenVanillaNN2(self.target, loadFromFile=True)
        elif typeOfGen == 4:         # GAN
            print("Generator: GenGAN")
            self.generator = GenGAN(self.target, loadFromFile=True)
        else:
            print("DanceDemo: typeOfGen error!!!")

    def draw(self):
        ske = Skeleton()
        image_err = np.zeros((256, 256, 3), dtype=np.uint8)
        image_err[:, :] = (0, 0, 255)  # (B, G, R)

        for i in range(self.source.getTotalFrames()):
            image_src = self.source.readFrame()

            if i % 5 == 0:
                isSke, image_src, ske = self.target.cropAndSke(image_src, ske)

                if isSke:
                    if (GEN_TYPE == 1 or GEN_TYPE == 4):
                        # Show the source image with the skeleton drawn on it
                        ske.draw(image_src)
                        image_tgt = self.generator.generate(ske)  # GENERATOR !!!
                        image_tgt = cv2.resize(image_tgt, (128, 128))
                    else:
                        # Show the skeleton vector on a blank image
                        skeleton_image = np.zeros((256, 256, 3), dtype=np.uint8)
                        ske.draw(skeleton_image)  # Draw the skeleton
                        image_tgt = self.generator.generate(ske)  # GENERATOR !!!
                        image_tgt = cv2.resize(image_tgt, (256, 256))
                else:
                    image_tgt = image_err

                # Combine the appropriate images
                image_combined = combineTwoImages(image_src if (GEN_TYPE == 1 or GEN_TYPE == 4) else skeleton_image, image_tgt)
                image_combined = cv2.resize(image_combined, (512, 256))

                # Show the combined image
                cv2.imshow('Image', image_combined)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                if key & 0xFF == ord('n'):
                    self.source.readNFrames(100)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    # NEAREST = 1
    # VANILLA_NN_SKE = 2
    # VANILLA_NN_Image = 3
    # GAN = 4
    GEN_TYPE = 1
    ddemo = DanceDemo("data/taichi2_full.mp4", GEN_TYPE)
    # ddemo = DanceDemo("data/taichi1.mp4")
    # ddemo = DanceDemo("data/karate1.mp4")
    ddemo.draw()