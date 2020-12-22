import numpy as np
import cv2
import os, argparse

class multiCamTriang(object):
    """docstring for multiCamTriang"""
    def __init__(self, track2Ds, poses, postition, Ks):
        super(multiCamTriang, self).__init__()
        self.track2Ds = track2Ds             # shape:(num_cam, num_frame, xy(2))
        self.poses = poses                   # shape:(num_cam, c2w(3, 3))
        self.postition = postition           # shape:(num_cam, xyz(3)) camera position in wcs
        self.Ks = Ks                         # shape:(num_cam, K(3,3))
        self.f = (Ks[:,0,0] + Ks[:,1,1]) / 2 # shape:(num_cam)
        self.p = Ks[:,0:2,2]                 # shape:(num_cam, xy(2))

        self.num_cam, self.num_frame, _ = self.track2Ds.shape

    def backProject(self):
        # Back project the 2d points of all frames to the 3d ray in world coordinate system

        # Shift origin to principal point
        self.track2Ds_ccs = self.track2Ds - self.p[:,None,:]

        # Back project 2D track to the CCS
        self.track2D_ccs = self.track2D_ccs / self.f[:,None,None]
        track_d = np.ones((self.num_cam, self.num_frame, 1))
        self.track2D_ccs = np.concatenate((self.track2D_ccs, track_d), axis=2)
        print("2D in CCS")
        pp.pprint(self.track2D_ccs)

        # 2D track described in WCS
        self.track2D_wcs = self.poses @ np.transpose(self.track2Ds, (0,2,1)) # shape:(num_cam, 3, num_frame)
        self.track2D_wcs = np.transpose(self.track2D_wcs, (0,2,1)) # shape:(num_cam, num_frame, 3)
        print("2D in WCS")
        pp.pprint(self.track2D_wcs)

    def getApprox3D(self):
        # Calculate the approximate solution of the ball postition by the least square method