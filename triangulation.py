import numpy as np
import cv2
import os, argparse
import pprint

from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import sqrt, pi, sin, cos, tan

from Hfinder import Hfinder
from tracker2D import tracker2D
from generator import startGL, readPose, toRad, toDeg, drawCourt, drawCircle, patternCircle, drawTrack

pp = pprint.PrettyPrinter(indent=4)

class multiCamTriang(object):
    """docstring for multiCamTriang"""
    def __init__(self, track2Ds, poses, eye, Ks):
        super(multiCamTriang, self).__init__()
        self.track2Ds = track2Ds             # shape:(num_cam, num_frame, xy(2)) 2D track from TrackNetV2
        self.poses = poses                   # shape:(num_cam, c2w(3, 3)) transform matrix from ccs to wcs
        self.eye = eye                       # shape:(num_cam, 1, xyz(3)) camera position in wcs
        self.Ks = Ks                         # shape:(num_cam, K(3,3)) intrinsic matrix
        self.f = (Ks[:,0,0] + Ks[:,1,1]) / 2 # shape:(num_cam) focal length
        self.p = Ks[:,0:2,2]                 # shape:(num_cam, xy(2)) principal point

        self.num_cam, self.num_frame, _ = self.track2Ds.shape
        self.backProject()
        self.getApprox3D()

    def backProject(self):
        # Back project the 2d points of all frames to the 3d ray in world coordinate system

        # Shift origin to principal point
        self.track2Ds_ccs = self.track2Ds - self.p[:,None,:]

        # Back project 2D track to the CCS
        self.track2Ds_ccs = self.track2Ds_ccs / self.f[:,None,None]
        track_d = np.ones((self.num_cam, self.num_frame, 1))
        self.track2Ds_ccs = np.concatenate((self.track2Ds_ccs, track_d), axis=2)
        print("2D in CCS")
        pp.pprint(self.track2Ds_ccs)

        # 2D track described in WCS
        self.track2D_wcs = self.poses @ np.transpose(self.track2Ds_ccs, (0,2,1)) # shape:(num_cam, 3, num_frame)
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
        print(self.track3D)
        '''
        [[-1.43517116  1.89159624  2.26335693]
         [-1.39877703  1.54092162  2.63527577]
         [-1.35540674  1.18507532  2.96552217]
         [-1.30677177  0.82912213  3.25820458]
         [-1.24912526  0.45949387  3.51172777]
         [-1.1845559   0.09085071  3.72145694]
         [-1.11731966 -0.28256041  3.89056493]
         [-1.03751997 -0.6640715   4.01358941]
         [-0.95234989 -1.04875004  4.09631244]
         [-0.85587677 -1.43730996  4.13223128]
         [-0.75232456 -1.83799202  4.12154178]
         [-0.63877272 -2.23848072  4.06290778]
         [-0.51448515 -2.6457232   3.95698365]
         [-0.38298821 -3.05165796  3.79782697]
         [-0.23923537 -3.46495259  3.59190024]
         [-0.08770912 -3.88709076  3.32807918]
         [ 0.07803184 -4.31217242  3.01206598]
         [ 0.25383237 -4.73946704  2.63708389]
         [ 0.44718799 -5.17209655  2.20506412]
         [ 0.65355034 -5.61191245  1.70384565]
         [ 0.87476952 -6.05531738  1.144601  ]]
        '''

if __name__ == '__main__':

    Kmtx = np.array([[1456.16303, 0, 1060/2],
                     [0, 1456.16303, 1920/2],
                     [0,          0,      1]])

    # Prepare Homography matrix (image(pixel) -> court(meter))
    poses = []
    eye = []
    Ks = []
    track2Ds = []
    img_list = ["synthetic_track/stereo/camera1/track_00000000.png",
                "synthetic_track/stereo/camera2/track_00000000.png"]
    for name in img_list:
        img = cv2.imread(name)
        court2D = []
        court3D = [[-3.05, 6.7], [3.05, 6.7], [3.05, -6.7], [-3.05, -6.7]]
        hf = Hfinder(img, court2D=court2D, court3D=court3D)
        Hmtx = hf.getH()

        tr2d = tracker2D(img)

        R = np.zeros((3,3))
        t = np.zeros(3)
        Rt = np.zeros((3,4))

        K_inv = np.linalg.inv(Kmtx)
        H_inv = np.linalg.inv(Hmtx) # H_inv: wcs -> ccs
        multiple = K_inv@H_inv[:,0]
        lamda = 1/np.linalg.norm(multiple, ord=None, axis=None, keepdims=False)
        
        R[:,0] = lamda*(K_inv@H_inv[:,0])
        R[:,1] = lamda*(K_inv@H_inv[:,1])
        R[:,2] = np.cross(R[:,0], R[:,1])
        t = np.array(lamda*(K_inv@H_inv[:,2]))

        Rt[:,:3] = R
        Rt[:,3] = t

        cir_position_ccs = (Rt @ [[0],[0],[0],[1]])
        cir_pose_i_ccs = (Rt @ [[1],[0],[0],[1]]) - cir_position_ccs
        cir_pose_j_ccs = (Rt @ [[0],[1],[0],[1]]) - cir_position_ccs
        cir_pose_k_ccs = (Rt @ [[0],[0],[1],[1]]) - cir_position_ccs
        c2w = np.array(
            [cir_pose_i_ccs.reshape(-1),
             cir_pose_j_ccs.reshape(-1),
             cir_pose_k_ccs.reshape(-1)]
        )
        cam_position_wcs = (c2w @ -cir_position_ccs)

        eye.append(cam_position_wcs.T)
        poses.append(c2w)
        Ks.append(Kmtx)
        if "camera2" in name:
            track2Ds.append(sorted(tr2d.getTrack2D(), key=lambda x:x[0], reverse=True))
        else:
            track2Ds.append(sorted(tr2d.getTrack2D(), key=lambda x:x[0]))

    mct = multiCamTriang(
        track2Ds=np.array(track2Ds), 
        poses=np.array(poses), 
        eye=np.array(eye), 
        Ks=np.array(Ks)
    )