import numpy as np
import cv2
import os, argparse

class multiCamTriang(object):
    """docstring for multiCamTriang"""
    def __init__(self, track2Ds, poses, Ks):
        super(multiCamTriang, self).__init__()
        self.track2Ds = track2Ds
        self.poses = poses
        self.Ks = Ks


