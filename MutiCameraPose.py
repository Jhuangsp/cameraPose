import numpy as np
import cv2

import CameraPose as cp

class MutiCameraPose(object):
    """docstring for MutiCameraPose"""
    def __init__(self, imgs, cmtxs, dists):
        super(MutiCameraPose, self).__init__()

        self.cameras_pose = [cp.CameraPose(img, cmtx_new, dist) for img, cmtx_new, dist in zip(imgs, cmtxs, dists)]


if __name__ == '__main__':
    
    imgl = cv2.imread('Left.png')
    imgr = cv2.imread('Right.png')

    cv_file_l = cv2.FileStorage("Left_cmx_dis.xml", cv2.FILE_STORAGE_READ)
    cv_file_r = cv2.FileStorage("Right_cmx_dis.xml", cv2.FILE_STORAGE_READ)
    cmtx_l = cv_file_l.getNode("intrinsic").mat()
    dist_l = cv_file_l.getNode("distortion").mat()
    cmtx_r = cv_file_r.getNode("intrinsic").mat()
    dist_r = cv_file_r.getNode("distortion").mat()
    print("Left Camera:")
    print("Intrinsic Matrix \n", cmtx_l)
    print("Distortion Coefficient \n", dist_l)
    print("Right Camera:")
    print("Intrinsic Matrix \n", cmtx_r)
    print("Distortion Coefficient \n", dist_r)
    cv_file_l.release()
    cv_file_r.release()
    cmtx_l_new = cmtx_l.copy()
    cmtx_r_new = cmtx_r.copy()
    map1, map2 = cv2.initUndistortRectifyMap(cmtx_l, dist_l, np.eye(3), cmtx_l_new, (imgl.shape[1], imgl.shape[0]), cv2.CV_16SC2)
    imgl = cv2.remap(imgl, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    map1, map2 = cv2.initUndistortRectifyMap(cmtx_r, dist_r, np.eye(3), cmtx_r_new, (imgr.shape[1], imgr.shape[0]), cv2.CV_16SC2)
    imgr = cv2.remap(imgr, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mutiCamSys = CameraPose([imgl, imgr], [cmtx_l_new, cmtx_r_new], [dist_l, dist_r])