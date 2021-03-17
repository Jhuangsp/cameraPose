import cv2
import numpy as np
import os, argparse
import pprint
import csv

from triangulation import multiCamTriang
from Hfinder import Hfinder
from Pfinder import Pfinder

pp = pprint.PrettyPrinter(indent=4)

if __name__ == '__main__':
    # Input videos
    # videos = ["Quadra/set3/2021-01-29_085648_LB.avi",
    #           "Quadra/set3/2021-01-29_085648_LF.avi",
    #           "Quadra/set3/2021-01-29_085648_RB.avi",
    #           "Quadra/set3/2021-01-29_085648_RF.avi"]
    videos = ["Quadra/set1/2021-01-29_084149_LF.avi",
              "Quadra/set1/2021-01-29_084149_RF.avi"]

    # TrackNetV2 results
    # track_files = ["Quadra/set3/2021-01-29_085648_LB_predict_denoise.csv",
    #                "Quadra/set3/2021-01-29_085648_LF_predict_denoise.csv",
    #                "Quadra/set3/2021-01-29_085648_RB_predict_denoise.csv",
    #                "Quadra/set3/2021-01-29_085648_RF_predict_denoise.csv"]
    track_files = ["Quadra/set1/2021-01-29_084149_LF_predict_denoise.csv",
                   "Quadra/set1/2021-01-29_084149_RF_predict_denoise.csv"]
    track2Ds = []
    masks = []
    for file in track_files:
        print(file)
        tmp_m = []
        tmp_t = []
        min_len = 1e10
        with open(file, newline='') as fs:
            shots = csv.DictReader(fs)
            for shot in shots:
                tmp_m.append(eval(shot['Visibility']) == 1)
                tmp_t.append(np.array([float(shot['X']), float(shot['Y'])],float))
            min_len = min(min_len, len(tmp_m))
            masks.append(tmp_m)
            track2Ds.append(np.array(tmp_t))
    min_len=100
    track2Ds = [i[:min_len] for i in track2Ds]


    # Read intrinsic
    # intrinsic_files = ["Quadra/param/intrinsicLB.xml",
    #                    "Quadra/param/intrinsicLF.xml",
    #                    "Quadra/param/intrinsicRB.xml",
    #                    "Quadra/param/intrinsicRF.xml"]
    intrinsic_files = ["Quadra/param/intrinsicLF.xml",
                       "Quadra/param/intrinsicRF.xml"]
    cmtx = []
    dist = []
    for file in intrinsic_files:
        fs = cv2.FileStorage(file, cv2.FILE_STORAGE_READ)
        cmtxNode = fs.getNode("intrinsic")
        distNode = fs.getNode("distortion")
        cmtx.append(cmtxNode.mat())
        dist.append(distNode.mat())
    fs.release()

    # Calculate Pose from Homography Matrix
    Ps = []
    C2Ws = []
    Eyes = []
#     [[ 1.96747713e-03  1.18591755e-02 -1.09144878e+01]
#  [ 1.18186490e-03 -1.30612518e-02  7.94603761e+00]
#  [-6.00825718e-05 -1.72094950e-03  1.00000000e+00]]
# [[ 1.48432957e-03 -7.65968011e-03  5.51229614e+00]
#  [-1.18005560e-03 -1.21379982e-02  1.14176017e+01]
#  [ 5.64985350e-06 -1.49136683e-03  1.00000000e+00]]
    for i, vid in enumerate(videos):
        # Get one frame from video
        cap = cv2.VideoCapture(vid)
        ret, frame = cap.read()
        if not ret:
            print("Reading video {} faild...".fromat(vid))
            os._exit(0)
        cap.release()

        # Calculate Homography matrix
        court2D = []
        court3D = [[-3.05, 6.7], [3.05, 6.7], [3.05, -6.7], [-3.05, -6.7]]
        hf = Hfinder(frame, court2D=court2D, court3D=court3D)
        Hmtx = hf.getH()
        print(Hmtx)

        # Calculate P and C2W matrix
        pf = Pfinder(cmtx[i], Hmtx)
        Ps.append(pf.getP())
        C2Ws.append(pf.getC2W())
        Eyes.append(pf.getCamera().T)

    # Triangulation
    pp.pprint(C2Ws)
    pp.pprint(Eyes)
    mct = multiCamTriang(
        track2Ds=np.array(track2Ds), 
        poses=np.array(C2Ws), 
        eye=np.array(Eyes), 
        Ks=np.array(cmtx)
    )