import cv2
import numpy as np
import os
import glob
import argparse
import yaml
import math

from CameraCalibration.camera_calibration import findChess

class CameraPoseChess(object):
    """docstring for CameraPoseChess"""
    def __init__(self, img, intrinsic, shape, size):
        super(CameraPoseChess, self).__init__()
        self.img = img
        self.K = intrinsic
        self.chess_w = shape[0]
        self.chess_h = shape[1]
        self.cube_size = size

        self.getExtrinsics()

    def getExtrinsics(self):
        objpoints = np.zeros((self.chess_w*self.chess_h,3), np.float32)
        objpoints[:,:2] = np.mgrid[0:self.chess_w, 0:self.chess_h].T.reshape(-1,2)
        objpoints[:,0] = objpoints[:,0] - self.chess_w//2
        objpoints[:,1] = objpoints[:,1] - self.chess_h//2

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # find chessboard
        ret, corners = cv2.findChessboardCorners(gray, (self.chess_w,self.chess_h), None)
        # print(corners.shape)
        if ret == True:
            self.H_mtx, mask = cv2.findHomography(objpoints[:,:2], corners, cv2.RANSAC, 5.0)
            self.R_mtxs = np.zeros((3, 3), np.float64)
            self.T_mtxs = np.zeros((3), np.float64)

            K_inv = np.linalg.inv(self.K)
            multiple = K_inv@self.H_mtx[:,0]
            lamda = 1/np.linalg.norm(multiple, ord=None, axis=None, keepdims=False)
            self.R_mtxs[:,0] = np.array(lamda*(K_inv@self.H_mtx[:,0]))
            self.R_mtxs[:,1] = np.array(lamda*(K_inv@self.H_mtx[:,1]))
            self.R_mtxs[:,2] = np.cross(self.R_mtxs[:,0], self.R_mtxs[:,1])
            self.T_mtxs = np.array(lamda*(K_inv@self.H_mtx[:,2])) * self.cube_size

            print(cv2.Rodrigues(self.R_mtxs)[0] * 180 / math.pi)
            print((self.R_mtxs @ np.eye(3)).T)
            print(self.R_mtxs)
            print(self.T_mtxs)

            # World Coordinate System (described in the camera coordinate system)
            self.O = self.T_mtxs
            pose = (self.R_mtxs @ np.eye(3)).T
            self.i = pose[0] if pose[0][0] >=0 else -pose[0]
            self.j = pose[1] if pose[0][0] >=0 else -pose[1]
            self.k = pose[2]            

            # Camera Coordinate System (described in the world coordinate system)
            self.Optical_center = np.array([self.i, self.j, self.k]) @ (np.array([0,0,0]) - self.O).reshape(3,1)
            self.i_c = np.array([self.i, self.j, self.k]) @ np.array([1,0,0]).reshape(3,1)
            self.j_c = np.array([self.i, self.j, self.k]) @ np.array([0,1,0]).reshape(3,1)
            self.k_c = np.array([self.i, self.j, self.k]) @ np.array([0,0,1]).reshape(3,1)
        pass

    def getH(self):
        return self.H_mtx

    def getCameraPosition(self):
        try:
            return self.Optical_center
        except:
            return None

    def getCameraPose(self):
        try:
            return (self.i_c, self.j_c, self.k_c)
        except:
            return None

    def getChessPosition(self):
        try:
            return self.O
        except:
            return None

    def getChessPose(self):
        try:
            return (self.i, self.j, self.k)
        except:
            return None

    def calib(self, imgs):
        objpoints, imgpoints = findChess(self.chess_w,self.chess_h,imgs)
        objpoints = np.array(objpoints)
        objpoints[:,:,0] = objpoints[:,:,0] - self.chess_w//2
        objpoints[:,:,1] = objpoints[:,:,1] - self.chess_h//2
        print('Camera calibration...')
        imgshape = cv2.imread(imgs[0]).shape
        img_size = (imgshape[1], imgshape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        tvecs = [t * self.cube_size for t in tvecs]
        print(mtx)
        print()
        print(rvecs)
        print()
        print(tvecs)
        Vr = np.array(rvecs)
        Tr = np.array(tvecs) 
        extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Pose Chessboard Estimation')
    parser.add_argument('--infolder', '-i', type=str, required=True,
                        help='path to input image files')
    parser.add_argument('--outfolder', '-o', type=str, required=True,
                        help='path to store output files')
    parser.add_argument('--exten', type=str, default='png',
                        help='input files extension')

    args = parser.parse_args()
    print(args)

    infolder = args.infolder
    outfolder = args.outfolder
    images = glob.glob(os.path.join(infolder, '*.' + args.exten))
    labels = glob.glob(os.path.join(infolder, 'labels', '*.yml'))
    for image, label in zip(images[:], labels[:]):
        name = image.split('.')
        name.pop(-1)
        name = '.'.join(name)
        print('------------ Starting', name, '------------')
        img = cv2.imread(image)
        with open(label, 'r') as fy:
            data = yaml.load(fy, Loader=yaml.FullLoader)
        assert data['pattern'] == 'chessboard', 'The label pattern must be \'chessboard\' not {}'.format(data['pattern'])
        cmtx_new = np.zeros((3,3))
        cmtx_new[0,0] = data['intrinsic']['focal']
        cmtx_new[1,1] = data['intrinsic']['focal']
        cmtx_new[2,2] = 1
        cmtx_new[0,2] = data['intrinsic']['w'] // 2
        cmtx_new[1,2] = data['intrinsic']['h'] // 2
        name = outfolder + os.sep + name.split(os.sep)[-1]

        cpc = CameraPoseChess(img, cmtx_new, data['chess'], data['cube_size'])

        cam_position_spcs = cpc.getCameraPosition()
        cam_pose_spcs = cpc.getCameraPose()
        che_position_ccs = cpc.getChessPosition()
        che_pose_ccs = cpc.getChessPose()

        if not os.path.isdir(name):
            os.makedirs(name)
        fs = cv2.FileStorage(name + "/pose.yml", cv2.FILE_STORAGE_WRITE)

        if type(cam_position_spcs) == type(None) or type(cam_pose_spcs) == type(None) or type(che_position_ccs) == type(None) or type(che_pose_ccs) == type(None):
            fs.write('success', 0)
            print()
            print('No chessboard detected .... skip')
        else:
            fs.write('success', 1)
            print()
            print('Camera position in SPCS:\n', cam_position_spcs)
            print('Camera pose:')
            print(cam_pose_spcs[0])
            print(cam_pose_spcs[1])
            print(cam_pose_spcs[2])
            print()
            print('Chess position in CCS:\n', che_position_ccs)
            print('Chess pose:')
            print(che_pose_ccs[0])
            print(che_pose_ccs[1])
            print(che_pose_ccs[2])

            fs.write('cam_position_spcs', cam_position_spcs)
            fs.write('cam_pose_i_spcs', cam_pose_spcs[0])
            fs.write('cam_pose_j_spcs', cam_pose_spcs[1])
            fs.write('cam_pose_k_spcs', cam_pose_spcs[2])
            fs.write('che_position_ccs', che_position_ccs)
            fs.write('che_pose_i_ccs', che_pose_ccs[0])
            fs.write('che_pose_j_ccs', che_pose_ccs[1])
            fs.write('che_pose_k_ccs', che_pose_ccs[2])

        fs.release()