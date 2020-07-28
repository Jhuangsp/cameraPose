import cv2
import numpy as np
import math
import time
import os, glob

class EllipsesDetection():
    """docstring for EllipsesDetection"""
    def __init__(self, img, intrinsic, distortion, radius=1, saveImgs=None):
        super(EllipsesDetection, self).__init__()

        self.imgC = img
        self.saveImgs = saveImgs
        self.intrinsic = intrinsic
        self.distortion = distortion
        self.radius = radius
        self.ellipses_ = []
        self.ori_h, self.ori_w, _ = img.shape
        if self.saveImgs != None:
            self.draw = np.zeros((img.shape[0]*2, img.shape[1]*2, 3), 'uint8')
            if not os.path.isdir(self.saveImgs):
                os.makedirs(self.saveImgs)
            else:
                for f in glob.glob(os.path.join(self.saveImgs, '*')):
                    os.remove(f)

        self.findEllipses()
        self.estimatePose()

    class ObliqueCone(object):
        """docstring for ObliqueCone"""
        def __init__(self):
            super(EllipsesDetection.ObliqueCone, self).__init__()
            self.Q = np.zeros((3,3))    # 3d ellipse corn equation
            self.Q_2d = np.zeros((3,3)) # 2d ellipse equation
            self.translations = np.zeros((2,3)) # two plausible translations solutions
            self.normals = np.zeros((2,3)) # two plausible translations solutions
            self.projections = np.zeros((2,2)) # two plausible projection of center of circle
            self.R = np.zeros((2,3,3)) # two plausible rotations solutions

        def set(self, box):
            a, b = box['axis']
            u, v = box['center']
            angle = math.pi * box['theta'] / 180.0
            ca = math.cos(angle)
            sa = math.sin(angle)
            cx = box['imgSize'][0]/2
            cy = box['imgSize'][1]/2
            Re = np.array([[ca, -sa], 
                           [sa,  ca]])
            ABInvTAB = np.array([[1./(a*a), 0.], 
                                 [0., 1./(b*b)]]);
            X0 = np.array([u-cx, v-cy])
            M = Re @ ABInvTAB @ Re.T
            Mf = X0.T @ M @ X0;
            A = M[0,0];
            B = M[0,1];
            C = M[1,1];
            D = - A * X0[0] - B * X0[1];
            E = - B * X0[0] - C * X0[1];
            F = Mf - 1.0;

            self.Q_2d = np.array([[A, B, D],
                                  [B, C, E],
                                  [D, E, F]]);

        def pose(self, intrinsic, radius):
            fx = intrinsic[0,0]
            fy = intrinsic[1,1]
            f = (fx + fy) / 2.0;
            self.Q = self.Q_2d * np.array([[1,   1,   1/f],
                                           [1,   1,   1/f], 
                                           [1/f, 1/f, 1/(f*f)]])

            ret, E, V = cv2.eigen(self.Q)
            V = V.T
            e1 = E[0,0]
            e2 = E[1,0]
            e3 = E[2,0]
            S1 = [+1,+1,+1,+1,-1,-1,-1,-1]
            S2 = [+1,+1,-1,-1,+1,+1,-1,-1]
            S3 = [+1,-1,+1,-1,+1,-1,+1,-1]
            g = math.sqrt((e2-e3)/(e1-e3))
            h = math.sqrt((e1-e2)/(e1-e3))

            k = 0
            for i in range(8):
                z0 =  S3[i] * (e2 * radius) / math.sqrt(-e1*e3);

                # Rotated center vector
                Tx =  S2[i] * e3/e2 * h
                Ty =  0.
                Tz = -S1[i] * e1/e2 * g

                # Rotated normal vector
                Nx =  S2[i] * h
                Ny =  0.
                Nz = -S1[i] * g

                t = z0 * V @ np.array([Tx, Ty, Tz]) # Center of circle in CCS
                n =      V @ np.array([Nx, Ny, Nz]) # Normal vector unit in CCS

                # identify the two possible solutions
                if (t[2] > 0) and (n[2] < 0): # Check constrain
                    if k > 1: continue
                    self.translations[k] = t
                    self.normals[k] = n
                    
                    # Projection
                    Pc = intrinsic @ t
                    self.projections[k,0] = Pc[0]/Pc[2];
                    self.projections[k,1] = Pc[1]/Pc[2];
                    k += 1
            pass

        def rotation2Normal(self, i):
            pass

        def normal2Rotation(self, i):
            unitZ = np.array([0, 0, 1]);
            nvec = np.copy(self.normals[i]);
            nvec = nvec/cv2.norm(nvec);
            c2 = nvec;
            c1 = np.cross(unitZ, c2);
            c0 = np.cross(c1, c2);
            c1 = c1/cv2.norm(c1);
            c0 = c0/cv2.norm(c0);
            self.R[i] = np.array([[c0[0], c1[0], c2[0]], 
                                  [c0[1], c1[1], c2[1]], 
                                  [c0[2], c1[2], c2[2]]]);
            pass

    class Ellipse(object):
        """docstring for Ellipse"""
        def __init__(self, lengthOfAxis, center, theta, imgSize):
            super(EllipsesDetection.Ellipse, self).__init__()
            self.lengthOfAxis = lengthOfAxis
            self.center = center
            self.theta = theta
            self.imgSize = imgSize
            self.boxEllipses = {
                 'axis':self.lengthOfAxis,
                 'center':self.center,
                 'theta':self.theta,
                 'imgSize':self.imgSize
            }
            self.cone = EllipsesDetection.ObliqueCone()

    class Marker(object):
        """docstring for Marker"""
        def __init__(self):
            super(EllipsesDetection.Marker, self).__init__()
            

    def findEllipses(self):
        t1 = time.time()
        self.edge_detection()
        t2 = time.time()
        print('Done Edge detection {} (sec)'.format(t2 - t1))
        self.contour_detection()
        t1 = time.time()
        print('Done Contour detection {} (sec)'.format(t1 - t2))
        self.hough_ellipses()
        t2 = time.time()
        print('Done Ellipses detection {} (sec)'.format(t2 - t1))
        pass

    def edge_detection(self):
        gray = cv2.cvtColor(self.imgC, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (3, 3), 0).astype(int)
        self.edges = cv2.Canny(gray.astype('uint8'), 85, 85*3, apertureSize=3)

    def contour_detection(self):
        self.contours, self.hierarchy = cv2.findContours(self.edges, 
            cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    def hough_ellipses(self):
        for i, cnt in enumerate(self.contours):
            if len(cnt) > (self.ori_h+self.ori_w) / 7 and len(cnt) < (self.ori_h+self.ori_w) * 6 / 7:
                print('contours #{}'.format(i))
                cnt = cnt.reshape(-1,2)
                x_g, y_g = np.round(cnt.sum(0)/len(cnt)).astype(int)

                # setup center and a with geometric constrain
                x = np.arange(self.ori_w)
                y = np.arange(self.ori_h)
                xv, yv = np.meshgrid(x,y)
                max_distance = np.zeros((self.ori_h, self.ori_w), int)
                max_st = time.time()
                # for pt in cnt:
                #     distance = np.sqrt((xv - pt[0])**2 + (yv - pt[1])**2)
                #     max_distance = np.maximum(max_distance, distance)
                # a = np.min(max_distance)
                # center_x, center_y = np.where(max_distance == a)

                patch_size = 100
                max_distance_patch = np.zeros((patch_size*2, patch_size*2), int)
                xv_p = np.expand_dims(xv[y_g-patch_size:y_g+patch_size,x_g-patch_size:x_g+patch_size], axis=2).repeat(len(cnt), axis=2)
                yv_p = np.expand_dims(yv[y_g-patch_size:y_g+patch_size,x_g-patch_size:x_g+patch_size], axis=2).repeat(len(cnt), axis=2)
                distance = np.sqrt((xv_p - cnt[:,0])**2 + (yv_p - cnt[:,1])**2)
                max_distance_patch = distance.max(axis=2)
                assert max_distance_patch.shape == (patch_size*2, patch_size*2), 'max_distance_patch.shape = {}'.format(max_distance_patch.shape)
                a = np.min(max_distance_patch)
                center_px, center_py = np.where(max_distance_patch == a)
                center_y = xv_p[center_px, center_py, 0]
                center_x = yv_p[center_px, center_py, 0]
                max_distance[y_g-patch_size:y_g+patch_size,x_g-patch_size:x_g+patch_size] = max_distance_patch

                # ind = np.argpartition(max_distance.reshape(-1), 2)[:2]
                # center_x, center_y = np.unravel_index(ind, max_distance.shape)
                # print(np.array([center_x,center_y]))
                # center = (center_x.sum()/len(center_x), center_y.sum()/len(center_y))

                print("Max:", time.time() - max_st)
                print(np.array([center_x,center_y]))
                center = (center_x.sum()/len(center_x), center_y.sum()/len(center_y))
                print(center, a)

                tmp_img = np.copy(self.imgC)
                tmp_cnt = np.zeros_like(tmp_img)
                if self.saveImgs != None:
                    cv2.drawContours(tmp_img,[cnt],-1,(0,0,255),1)
                    cv2.drawContours(tmp_cnt,[cnt],-1,(255,255,255),1)
                    cv2.circle(tmp_img,(round(center[1]), round(center[0])), 5, (0,0,255), 1)
                    cv2.circle(tmp_cnt,(round(center[1]), round(center[0])), 5, (0,0,255), 1)
                    self.draw[:self.ori_h, :self.ori_w] = tmp_img
                    self.draw[self.ori_h:, :self.ori_w] = tmp_cnt
                    self.draw[:self.ori_h, self.ori_w:] = cv2.cvtColor(((max_distance/np.max(max_distance))*255).astype('uint8'), cv2.COLOR_GRAY2BGR)
                    self.draw[self.ori_h:, self.ori_w:] = [255, 255, 255]
                
                # hough transform and voting
                hough_st = time.time()
                hough_space = np.zeros((int(a+1), 180), int)
                for pt in cnt:
                    for w in range(0,180): # theta
                        G = w * math.pi / 180
                        XX = ((pt[0]-center[1])*math.cos(G)+(pt[1]-center[0])*math.sin(G))**2/(a**2)
                        YY = (-(pt[0]-center[1])*math.sin(G)+(pt[1]-center[0])*math.cos(G))**2
                        B = round(math.sqrt(abs(YY/(1-XX)))+1)
                        if B > 0 and B <= a:
                             hough_space[B,w] = hough_space[B,w]+1
                print("Hough:", time.time() - hough_st)

                # G = np.arange(0,180) * math.pi / 180
                # XX = ((cnt[:,0] - center[1]).reshape(1,-1).T @ np.cos(G).reshape(1,-1) + (cnt[:,1] - center[0]).reshape(1,-1).T @ np.sin(G).reshape(1,-1))**2 / (a**2)
                # YY = (-(cnt[:,0] - center[1]).reshape(1,-1).T @ np.sin(G).reshape(1,-1) + (cnt[:,1] - center[0]).reshape(1,-1).T @ np.cos(G).reshape(1,-1))**2
                # B = np.round(np.sqrt(np.abs(YY/(1-XX)))+1)
                # vote_mask = np.logical_and((B > 0), (B <= a))
                # B = B*vote_mask
                # cns, ws = np.where(B > 0)
                # for c,w in zip(cns,ws):
                #     hough_space[B[c,w],w] = hough_space[B,w]+1

                max_para = hough_space.max()
                b, w = np.where(hough_space > max_para - len(cnt)/10) # select group of answers rather than just one
                max_para = hough_space[b,w]
                max_para = max_para.sum()/len(max_para)
                max_para = max_para.sum()#/len(max_para)
                bb = b.sum()/len(b) # avg b
                ww = w.sum()/len(w) # avg theta
                
                if max_para <= len(cnt)*0.15 or a/bb > 2.5:
                    print('[No result] {}/{} < 0.2, a/b {} > 2.5'.format(max_para, len(cnt), a/bb))
                else:
                    print('[Detected] b = {}, theta={}'.format(bb, ww*math.pi/180))
                    self.ellipses_.append(EllipsesDetection.Ellipse((a, bb), (center[1], center[0]), ww, (self.ori_w, self.ori_h)))
                    
                    if self.saveImgs != None:
                        tmp = np.copy(self.imgC)
                        tmp = cv2.ellipse(tmp, ((center[1], center[0]), (a*2, bb*2), ww), (0, 255, 0), 1)
                        tmp_cnt = cv2.ellipse(tmp_cnt, ((center[1], center[0]), (a*2, bb*2), ww), (0, 255, 0), 1)
                        self.draw[self.ori_h:, :self.ori_w] = tmp_cnt
                        self.draw[self.ori_h:, self.ori_w:] = tmp
                        # cv2.imshow('result', tmp)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                
                if self.saveImgs != None:
                    hough_space = ((hough_space/hough_space.max()) * 255).astype('uint8')
                    hough_space = cv2.cvtColor(hough_space, cv2.COLOR_GRAY2BGR)
                    hough_space[b,w] = [0,0,255]
                    hough_space = cv2.resize(hough_space.astype('uint8'), (hough_space.shape[1]*2, hough_space.shape[0]*2), interpolation=cv2.INTER_NEAREST)
                    self.draw[-hough_space.shape[0]:, self.ori_w:self.ori_w+hough_space.shape[1]] = hough_space
                    cv2.imwrite(os.path.join(self.saveImgs, 'draw_{}.png'.format(i)), self.draw)
                    self.draw = np.zeros((self.ori_h*2, self.ori_w*2, 3), 'uint8')
                print('')

    def estimatePose(self):
        for ellipses in self.ellipses_:
            ellipses.cone.set(ellipses.boxEllipses)
            ellipses.cone.pose(self.intrinsic, self.radius)
            ellipses.cone.normal2Rotation(0);
            ellipses.cone.normal2Rotation(1);
            print(ellipses.cone.R[0])
            print(ellipses.cone.R[1])
        pass


if __name__ == '__main__':
    # img = cv2.imread('e1.jpg')  
    # img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
    img = cv2.imread('Left.png')

    cv_file = cv2.FileStorage("Left_cmx_dis.xml", cv2.FILE_STORAGE_READ)
    cmtx = cv_file.getNode("intrinsic").mat()
    dist = cv_file.getNode("distortion").mat()
    print("Intrinsic Matrix \n", cmtx)
    print("Distortion Coefficient \n", dist)
    cv_file.release()
    cmtx_new = cmtx.copy()
    map1, map2 = cv2.initUndistortRectifyMap(cmtx, dist, np.eye(3), cmtx_new, (img.shape[1], img.shape[0]), cv2.CV_16SC2)
    img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    detectionNode = EllipsesDetection(img, cmtx_new, dist, radius=5, saveImgs='result2')