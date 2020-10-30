import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt
import argparse
import yaml, json
import math
import pandas as pd
import seaborn as sns

def deg2rad(deg):
    return deg * math.pi / 180

def rad2deg(rad):
    return rad * 180 / math.pi

def mse(v1, v2):
    v = v1 - v2
    return np.linalg.norm(v, axis=-1)

def cosErr(v1, v2):
    nv1 = np.linalg.norm(v1, axis=-1)
    nv2 = np.linalg.norm(v2, axis=-1)
    costheta = (v1*v2).sum(-1) / (nv1*nv2)
    return np.arccos(costheta)

class Evaluator(object):
    """docstring for Evaluator"""

    def __init__(self, conf, labels, results, out, maskfile=None, infofile=None, loadnpy=False):
        super(Evaluator, self).__init__()
        self.labels = labels
        self.results = results
        self.out = out
        self.maskfile = maskfile
        self.infofile = infofile
        self.loadnpy = loadnpy

        fconf = open(conf)
        self.p_cfg = json.load(fconf)
        fconf.close()

        # get config info
        self.c_patt = self.p_cfg['pattern']
        self.eval()

    def eval(self):
        if self.loadnpy:
            print("loading labels & results from .npy")
            self.gt = np.load(self.labels)
            self.re = np.load(self.results)
            self.info = np.load(self.infofile)
            self.mask = np.load(self.maskfile)
        else:
            self.loaddata()
            self.saveLabel()
            self.saveResult()

        self.invalid_rate = np.logical_not(self.mask).sum() / self.mask.size

        self.fig = plt.figure(figsize=(6,6))

        # pitch matter
        self.pitchMatter()

        plt.show()

    def pitchMatter(self):
        print("performing pitch matter evaluation...")
        print("section size: {} (degree)".format(5))
        # each section has 5 degree
        section = deg2rad(5)

        # calculate pitch
        re_p = self.info[:, 0]
        re_p = (re_p // section).astype('int')
        re_px = np.copy(re_p)
        re_px[np.logical_not(self.mask)] = -1

        # calculate error
        self.p_counts = []
        self.p_success = []
        self.p_terrs = []
        self.p_rerrs_i = []
        self.p_rerrs_j = []
        self.p_rerrs_k = []
        for i in range(re_p.min(), re_p.max()+1):
            print("section #{} ...".format(i))

            # mask of selected condition and its success samples
            sec_mask = re_p == i
            sec_mask_succ = re_px == i

            # samples that match the selected condition and its success rate
            self.p_counts.append(sec_mask.sum()) 
            self.p_success.append(sec_mask_succ.sum()/sec_mask.sum()) 

            if self.p_success[-1] > 0:
                sec_gt = self.gt[sec_mask_succ]
                sec_re = self.re[sec_mask_succ]

                # change cv2's detection of chessboard to our defined coordination
                sec_re = sec_re if self.c_patt == 'two_circle' else sec_re * \
                    np.array([1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1])

                terr = mse(sec_re[:,0:3], sec_gt[:,0:3])
                rerrs_i = cosErr(sec_re[:,3:6],  sec_gt[:,3:6])
                rerrs_j = cosErr(sec_re[:,6:9],  sec_gt[:,6:9])
                rerrs_k = cosErr(sec_re[:,9:12], sec_gt[:,9:12])
                
                self.p_terrs.append(terr)# / self.p_success[i])
                self.p_rerrs_i.append(rerrs_i)# / self.p_success[i])
                self.p_rerrs_j.append(rerrs_j)# / self.p_success[i])
                self.p_rerrs_k.append(rerrs_k)# / self.p_success[i])
            else:
                self.p_terrs.append([np.nan, np.nan, np.nan, np.nan, np.nan])
                self.p_rerrs_i.append([np.nan, np.nan, np.nan, np.nan, np.nan])
                self.p_rerrs_j.append([np.nan, np.nan, np.nan, np.nan, np.nan])
                self.p_rerrs_k.append([np.nan, np.nan, np.nan, np.nan, np.nan])
        self.showPitchMatter()

    def showPitchMatter(self):
        print("preparing the result")
        # Add one subplot
        na = len(self.fig.axes)
        for i in range(na):
            self.fig.axes[i].change_geometry(1, na+1, i+1)

        section = deg2rad(5)
        re_p = self.info[:, 0]
        re_p = (re_p // section).astype('int')

        n_sec = [len(i) for i in self.p_rerrs_i]
        p_terrs = np.concatenate(self.p_terrs)
        p_rerrs_i = np.concatenate(self.p_rerrs_i)
        p_rerrs_j = np.concatenate(self.p_rerrs_j)
        p_rerrs_k = np.concatenate(self.p_rerrs_k)


        pit_label = []
        for i,n in enumerate(range(re_p.min(), re_p.max()+1)):
            for nn in range(n_sec[i]):
                pit_label.append("{:.3f} - {:.3f}".format(n*section, (n+1)*section))
        # pit_label = ["{:.3f} - {:.3f}".format(i*section, (i+1)*section) for i in range(re_p.min(), re_p.max()+1)]
        type_label = ["Translation Error (m)", "Rotation Error (rad)"]
        tra_label = ["camera position"]
        rot_label = ["axis_i", "axis_j", "axis_k"]

        # create DataFrame for rotation error
        rot_labels = pd.MultiIndex.from_product([[type_label[1]], rot_label, pit_label],
            names=['type', 'station', 'measurement'])
        rot_data = pd.DataFrame({'value': np.concatenate((p_rerrs_i, p_rerrs_j, p_rerrs_k))}, index=rot_labels)
        rot_data = rot_data.reset_index()

        # create DataFrame for translation error
        tra_labels = pd.MultiIndex.from_product([[type_label[0]], tra_label, pit_label],
            names=['type', 'station', 'measurement'])
        tra_data = pd.DataFrame({'value': p_terrs}, index=tra_labels)
        tra_data = tra_data.reset_index()

        # plot rotation error
        g1 = sns.FacetGrid(rot_data, col="measurement", row="type", height=3, aspect=2, sharey=True, sharex=True)
        g1.map(sns.boxplot, "station", "value", palette=['lightcoral', 'lightgreen', 'lightblue'])
        g1.set_axis_labels("", "val").set_titles("{col_name}").despine(bottom=True, left=True)
        g1.axes[0,0].set_ylabel(type_label[1])
        g1.set(ylim=(0, math.pi/2))
        g1.fig.suptitle('Pitch Matter Rotation error', y=0.99)
        g1.fig.set_size_inches(15, 4)
        for ax in g1.axes:
            for a in ax:
                a.grid(True)
        g1.savefig(os.path.join(self.out, "rotErr.png"))

        # plot translation error
        g2 = sns.FacetGrid(tra_data, col="measurement", row="type", height=3, aspect=1, sharey=True, sharex=True)
        g2.map(sns.boxplot, "station", "value", palette=['orange'])
        g2.set_axis_labels("", "val").set_titles("{col_name}").despine(bottom=True, left=True)
        g2.axes[0,0].set_ylabel(type_label[0])
        g2.set(ylim=(0, 30))
        g2.fig.suptitle('Pitch Matter Translation error', y=0.99)
        g2.fig.set_size_inches(15, 4)
        for ax in g2.axes:
            for a in ax:
                a.grid(True)
        g2.savefig(os.path.join(self.out, "traErr.png"))

        # Plot success rate
        self.ax = self.fig.add_subplot(1, na+1, na+1)
        self.ax.set_title("Pitch Matter Detection ({})".format(self.c_patt))
        self.ax.set_xlabel("Pitch (rad)")
        self.ax.set_ylabel("Success rate")
        self.ax.bar(np.arange(len(self.p_success)), self.p_success, 0.35, label='success rate')
        self.ax.set_xticks(np.arange(len(self.p_success)))
        self.ax.set_xticklabels(pit_label, rotation=45)
        self.ax.legend(loc="best")
        self.fig.tight_layout()
        self.fig.savefig(os.path.join(self.out, "succ.png"))

    def loaddata(self):
        print("loading labels & results...")
        assert len(self.labels) == len(self.results), 'len(self.labels) != len(self.results)'

        self.samples = len(self.labels)
        # shape: (samples, (pitch, tx, ty, distance, size, space))
        self.info = np.zeros((self.samples, 6))
        # shape: (samples, (trans(3), rotation(9)))
        self.gt = np.zeros((self.samples, 3+9))
        self.re = np.zeros((self.samples, 3+9))
        self.mask = np.zeros((self.samples), bool)

        for i, (l, r) in enumerate(zip(self.labels, self.results)):
            # Read label file
            with open(l, 'r') as fl:
                label_data = yaml.load(fl, Loader=yaml.FullLoader)

            # Load infomation
            self.info[i][0] = label_data['extrinsic']['rotation'][2]
            self.info[i][1] = label_data['extrinsic']['translation'][0]
            self.info[i][2] = label_data['extrinsic']['translation'][1]
            self.info[i][3] = label_data['extrinsic']['init_distance']
            if self.c_patt == "two_circle":
                self.info[i][4] = label_data['radius']
                self.info[i][5] = label_data['circle_shift']
            elif self.c_patt == "chessboard":
                self.info[i][4] = label_data['cube_size']
                self.info[i][5] = 0

            # Load label
            self.gt[i][0:3] = label_data['spcs']['cam_center']
            self.gt[i][3:6] = label_data['spcs']['axis'][0]
            self.gt[i][6:9] = label_data['spcs']['axis'][1]
            self.gt[i][9:12] = label_data['spcs']['axis'][2]

            # Read result file
            fr = cv2.FileStorage(r, cv2.FILE_STORAGE_READ)

            # Check detection
            if fr.getNode('success').real() == 1.:
                # Load result
                self.mask[i] = True
                self.re[i][0:3] = fr.getNode('cam_position_spcs').mat().reshape(-1)
                self.re[i][3:6] = fr.getNode('cam_pose_i_spcs').mat().reshape(-1)
                self.re[i][6:9] = fr.getNode('cam_pose_j_spcs').mat().reshape(-1)
                self.re[i][9:12] = fr.getNode('cam_pose_k_spcs').mat().reshape(-1)
            else:
                # Faild detection
                self.mask[i] = False

    def saveLabel(self):
        print("saving labels to npy...")
        name = self.labels[0].split(os.sep)
        name = os.sep.join(name[:-1])
        name_gt = os.path.join(name, 'label.npy')
        with open(name_gt, 'wb') as f:
            np.save(f, self.gt)
        name_info = os.path.join(name, 'info.npy')
        with open(name_info, 'wb') as f:
            np.save(f, self.info)

    def saveResult(self):
        print("saving results to npy...")
        name = self.results[0].split(os.sep)
        name = os.sep.join(name[:-2])
        name_data = os.path.join(name, 'data.npy')
        name_mask = os.path.join(name, 'mask.npy')
        with open(name_data, 'wb') as f:
            np.save(f, self.re)
        with open(name_mask, 'wb') as f:
            np.save(f, self.mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Pose Estimation')
    parser.add_argument('--conf', type=str, required=True,
                        help='path to generate config file')
    parser.add_argument('--label', type=str, required=True,
                        help='path to input image files')
    parser.add_argument('--result', type=str, required=True,
                        help='path to pose estimation output files')
    parser.add_argument('--mask', type=str,
                        help='path to success estimation mask npy files')
    parser.add_argument('--info', type=str,
                        help='path to success label info npy files')
    parser.add_argument('--out', type=str, required=True,
                        help='path to save output figures')
    args = parser.parse_args()
    print(args)

    if args.label.endswith('npy') and args.result.endswith('npy') and args.mask != None:
        ev = Evaluator(args.conf, args.label, args.result, out=args.out, maskfile=args.mask, infofile=args.info, loadnpy=True)
    else:
        labels = glob.glob(os.path.join(args.label,  'label_*.yml'))
        results = glob.glob(os.path.join(args.result, '*'))
        results = [os.path.join(r, 'pose.yml') for r in results if os.path.isdir(r)]

        print('{} label files'.format(len(labels)))
        print('{} result files'.format(len(results)))
        ev = Evaluator(args.conf, labels, results, out=args.out)
