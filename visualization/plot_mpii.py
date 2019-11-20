# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Modified by Depu Meng (mdp@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cv2
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import math
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize COCO predictions')
    # general
    parser.add_argument('--image-path',
                        help='Path of COCO val images',
                        type=str,
                        default='data/mpii/images/'
                        )

    parser.add_argument('--save-path',
                        help="Path to save the visualizations",
                        type=str,
                        default='visualize/mpii/')

    parser.add_argument('--prediction',
                        help="Prediction file to visualize",
                        type=str,
                        required=True)

    parser.add_argument('--style',
                        help="Style of the visualization: chunhua style or xiaochu style or openpose style",
                        type=str,
                        default='chunhua')

    args = parser.parse_args()

    return args


"""
# pose track
link_pairs = [
    [0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [2, 3],
    [9, 10], [10, 11], [12, 13], [13, 14], [2, 8], [3, 9]
]
"""

# joint[0]->joint[1](color: joint[2])
# MPII

class ColorStyle:
    def __init__(self, color, link_pairs, point_color, ignore_id):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color
        self.ignore_id = ignore_id
        for i in range(len(self.color)):
            self.link_pairs[i].append(tuple(np.array(self.color[i])/255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i])/255.))


# XiaoChu 
# (R,G,B)
color1 = [(0,109,45),(49,163,84),(255,255,51),(228,26,28),(179,0,0),
         (255,255,51), (240,2,127), (0,0,255), (44,127,184), (255,255,51),
         (255,255,51), (254,153,41), (217,95,14)]

link_pairs1 = [
    [0, 1], [1, 2], [2, 12], [3, 4], [4, 5], [3, 13], [8, 9],
    [10, 11], [11,12], [12, 7], [7, 13], [13, 14], [14, 15],
]

point_color1 = [(0,109,45),(49,163,84),(255,255,51),
               (255,255,51),(228,26,28),(179,0,0),
               (255,255,51),(240,2,127), (240,2,127),
               (0,0,255), (44,127,184), (255,255,51),
               (255,255,51), (254,153,41), (217,95,14)]

ignore_id1 = [6]

xiaochu_style = ColorStyle(color1, link_pairs1, point_color1, ignore_id1)

# Chunhua
# (R,G,B)
color2 = [(252,176,243),(252,176,243),(252,176,243),
         (0,176,240), (0,176,240), (0,176,240),
         (165, 104, 210), (255,0,0),
         (255,255,0), (255,255,0), (255,255,0),
         (169, 209, 142),(169, 209, 142),(169, 209, 142)]

link_pairs2 = [
    [0, 1], [1, 2], [2, 6], [6,3], [3, 4], [4, 5], [6, 7], [8, 9],
    [10, 11], [11,12], [12, 8], [8, 13], [13, 14], [14, 15],
]

point_color2 = [(252,176,243),(252,176,243),(252,176,243),
               (0,176,240),(0,176,240),(0,176,240),
               (165, 104, 210), (165, 104, 210), (255,0,0), (255,0,0),
               (255,255,0),(255,255,0), (255,255,0),
               (169, 209, 142), (169, 209, 142), (169, 209, 142)]

ignore_id2 = []

chunhua_style = ColorStyle(color2, link_pairs2, point_color2, ignore_id2)

# OpenPose
# (R,G,B)
color3 = [(121,67,226),(74,87,226),(47,118,177),
         (163,61,204), (216,53,204), (211,48,121),
         (63, 214, 217), (177,24,21),
         (43,192,128), (83,224,91), (111,210,58),
         (220, 132, 72),(194, 169, 37),(172, 214, 69)]

link_pairs3 = [
    [0, 1], [1, 2], [2, 6], [6,3], [3, 4], [4, 5], [6, 8], [8, 9],
    [10, 11], [11,12], [12, 8], [8, 13], [13, 14], [14, 15],
]

point_color3 = [(121,67,226),(74,87,226),(74,87,226),
               (163,61,204),(216,53,204),(211,48,121),
               (63, 214, 217),(63, 214, 217), (177,24,21),
               (43,192,128), (83,224,91), (111,210,58),
               (220, 132, 72), (194, 169, 37), (172, 214, 69)]

ignore_id3 = [7]

openpose_style = ColorStyle(color3, link_pairs3, point_color3, ignore_id3)


"""
def map_joint_array(joints, ignore_id):
    new_joints = np.zeros((16,3))
    for i in range(joints.shape[1]):
        new_joints[i,2] = int(joints[0,i][2][0][0])
        if new_joints[i,2] in ignore_id:
            continue
        else:
            new_joints[i,2] = int(joints[0,i][2][0][0])
            new_joints[i,0] = int(joints[0,i][0][0,0])
            new_joints[i,1] = int(joints[0,i][1][0,0])
    return new_joints
"""

def map_joint_array(joints, ignore_id):
    new_joints = []
    for i in range(joints.shape[1]):
        if int(joints[0,i][2][0][0]) in ignore_id:
            continue
        else:
            joint = [int(joints[0,i][j][0,0]) for j in range(3)]
            new_joints.append(joint)
    return np.array(new_joints)


def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[1]):
        x = int(joints[0,i][0][0,0])
        y = int(joints[0,i][1][0,0])
        id = int(joints[0,i][2][0][0])
        joints_dict[id] = (x, y)
        
    return joints_dict

def plot_joints(image, joints):
    for id, pos in joints.items():
        cv2.circle(image, pos, 3, (0,255,0), 2)
    


if __name__ == '__main__':
    args = parse_args()
    save_path = args.save_path
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except Exception:
            print('Fail to make {}'.format(save_path))
    pred = loadmat(args.prediction)['pred']

    # change color style here
    if args.style == 'chunhua':
        color_style = chunhua_style
    elif args.style == 'xiaochu':
        color_style = xiaochu_style
    elif args.style == 'openpose':
        color_style = openpose_style

    link_pairs = color_style.link_pairs
    ignore_id = color_style.ignore_id
    ring_color = color_style.ring_color

    for i in range(0, 1000):
        if len(pred[0,i][1]) < 1 or pred[0,i][1][0,0] is None or len(pred[0,i][1][0,0]) < 3 or len(pred[0,i][1][0,0][2][0]) < 1:
            continue
        img_name = pred[0,i][0][0,0][0][0][:-4]
        print('id: ', i)
        img_file = args.image_path + pred[0,i][0][0,0][0][0]
        scale = pred[0,i][1][0,0][0][0,0]
        center = (pred[0,i][1][0,0][1][0,0][0][0,0],pred[0,i][1][0,0][1][0,0][1][0,0])
        joints = pred[0,i][1][0,0][2][0,0][0]
        joints_array = map_joint_array(joints, ignore_id)
        joints_dict = map_joint_dict(joints)
        
        data_numpy = cv2.imread(img_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        h = data_numpy.shape[0]
        w = data_numpy.shape[1]
        ref = np.min((h,w))
        fig = plt.figure(figsize=(w/100, h/100), dpi=100)
        ax = plt.subplot(1,1,1)
        bk = plt.imshow(data_numpy[:,:,::-1])
        bk.set_zorder(-1)
        
        # stick 
        for link_pair in link_pairs:
            if link_pair[0] in joints_dict \
                and link_pair[1] in joints_dict:
                line = mlines.Line2D(
                        np.array([joints_dict[link_pair[0]][0],joints_dict[link_pair[1]][0]]),
                        np.array([joints_dict[link_pair[0]][1],joints_dict[link_pair[1]][1]]),
                        ls='-', lw=ref/90, alpha=1, color=link_pair[2],)
                line.set_zorder(0)
                ax.add_line(line)
        
        # black ring
        for j in range(joints_array.shape[0]):
            circle = mpatches.Circle(tuple(joints_array[j,:2]), radius=ref/90, 
                                     ec='black', fc=ring_color[j], alpha=1, linewidth=ref/270)
            circle.set_zorder(1)
            ax.add_patch(circle)
        
        
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.axis('off')
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)        
        plt.margins(0,0)
        plt.savefig(save_path + 'id_' +str(i)+ '.pdf', format='pdf', bbox_inckes='tight', dpi=100)
        plt.savefig(save_path + 'id_' +str(i)+ '.png', format='png', bbox_inckes='tight', dpi=100)
        plt.close()
