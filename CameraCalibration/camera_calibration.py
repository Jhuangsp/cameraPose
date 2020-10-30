import numpy as np
import os
import cv2
import glob
import matplotlib.pyplot as plt
from . import camera_calibration_show_extrinsics as show
from PIL import Image


def findChess(w,h,images,showimage=False):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # (8,6) is for the given testing images.
    # If you use the another data (e.g. pictures you take by your smartphone), 
    # you need to set the corresponding numbers.
    corner_x = w
    corner_y = h
    objp = np.zeros((corner_x*corner_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    print('Start finding chessboard corners...')
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray)

        #Find the chessboard corners
        print('find the chessboard corners of',fname)
        ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)
        #print(corners, corners.shape); os._exit(0)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
            if showimage:
                plt.imshow(img)
                plt.show()

    return objpoints, imgpoints

if __name__ == '__main__':
    # Make a list of calibration images
    images = glob.glob('data/*.JPG')
    objpoints, imgpoints = findChess(7,7,images)
    print('Camera calibration...')
    img = cv2.imread(images[0])
    img_size = (img.shape[1], img.shape[0])
    # You need to comment these functions and write your calibration function from scratch.
    # Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
    # In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
    #"""
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    Vr = np.array(rvecs)
    Tr = np.array(tvecs)
    extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
    #"""

    """
    Write your code here
    """

    # TODO

    """
    """

    print('Intrinsic\n', mtx)
    print('Extrinsic\n', extrinsics)

    # show the camera extrinsics
    print('Show the camera extrinsics, figure by ourselves')
    # plot setting
    # You can modify it for better visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    # camera setting
    camera_matrix = mtx
    cam_width = 0.064/0.1
    cam_height = 0.032/0.1
    scale_focal = 1600
    # chess board setting
    board_width = 8
    board_height = 6
    square_size = 1
    # display
    # True -> fix board, moving cameras
    # False -> fix camera, moving boards
    min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                    scale_focal, extrinsics, board_width,
                                                    board_height, square_size, True)

    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]
    max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

    mid_x = (X_max+X_min) * 0.5
    mid_y = (Y_max+Y_min) * 0.5
    mid_z = (Z_max+Z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, 0)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    ax.set_title('Extrinsic Parameters Visualization')
    plt.show()
