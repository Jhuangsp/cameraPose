import numpy as np
import cv2
import os, argparse

class multiCamTriang(object):
    """docstring for multiCamTriang"""
    def __init__(self, track2Ds, poses, eye, Ks):
        super(multiCamTriang, self).__init__()
        self.track2Ds = track2Ds             # shape:(num_cam, num_frame, xy(2)) 2D track from TrackNetV2
        self.poses = poses                   # shape:(num_cam, c2w(3, 3)) transform matrix from ccs to wcs
        self.eye = eye[:,None,:]             # shape:(num_cam, 1, xyz(3)) camera position in wcs
        self.Ks = Ks                         # shape:(num_cam, K(3,3)) intrinsic matrix
        self.f = (Ks[:,0,0] + Ks[:,1,1]) / 2 # shape:(num_cam) focal length
        self.p = Ks[:,0:2,2]                 # shape:(num_cam, xy(2)) principal point

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
        self.track2D_wcs = self.track2D_wcs / np.linalg.norm(self.track2D_wcs, axis=2)[:,:,None]
        print("2D in WCS")
        pp.pprint(self.track2D_wcs)

    def getApprox3D(self):
        # Calculate the approximate solution of the ball postition by the least square method
        # n-lines intersection == 2n-planes intersection

        planeA = np.copy(self.track2D_wcs)
        planeA[:,:,0] = 0
        planeA[:,:,1] = -self.track2D_wcs[:,:,2]
        planeA[:,:,2] = self.track2D_wcs[:,:,1]

        # check norm == 0
        planeA_tmp = np.copy(self.track2D_wcs)
        planeA_tmp[:,:,0] = -self.track2D_wcs[:,:,2]
        planeA_tmp[:,:,1] = 0
        planeA_tmp[:,:,2] = self.track2D_wcs[:,:,0]
        mask = np.linalg.norm(planeA, axis=2)==0
        planeA[mask] = planeA_tmp[mask]

        # # check norm == 0
        # planeA_tmp = np.copy(self.track2D_wcs)
        # planeA_tmp[:,:,0] = -self.track2D_wcs[:,:,1]
        # planeA_tmp[:,:,1] = self.track2D_wcs[:,:,0]
        # planeA_tmp[:,:,2] = 0
        # mask = np.linalg.norm(planeA, axis=2)==0
        # planeA[mask] = planeA_tmp[mask]

        planeB = np.cross(self.track2D_wcs, planeA)

        Amtx = np.concatenate((planeA, planeB), axis=0) # shape:(2num_cam, num_frame, 3)
        b = np.concatenate((self.eye*planeA, self.eye*planeB), axis=0).sum(-1)[:,:,None] # shape:(2num_cam, num_frame, 1)

        Amtx = np.transpose(Amtx, (1,0,2)) # shape:(num_frame, 2num_cam, 3)
        b = np.transpose(b, (1,0,2)) # shape:(num_frame, 2num_cam, 1)

        left = np.transpose(Amtx, (0,2,1)) @ Amtx # shape:(num_frame, 3, 3)
        right = np.transpose(Amtx, (0,2,1)) @ b # shape:(num_frame, 3, 1)

        self.track3D = np.linalg.pinv(left) @ right # shape:(num_frame, 3, 1)
        self.track3D = self.track3D.reshape(-1,3)

