# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Modified by Depu Meng (mdp@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os


class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.color)):
            self.link_pairs[i].append(tuple(np.array(self.color[i])/255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i])/255.))
        
# Xiaochu Style
# (R,G,B)
color1 = [(179,0,0),(228,26,28),(255,255,51),
    (49,163,84), (0,109,45), (255,255,51),
    (240,2,127),(240,2,127),(240,2,127), (240,2,127), (240,2,127), 
    (217,95,14), (254,153,41),(255,255,51),
    (44,127,184),(0,0,255)]

link_pairs1 = [
        [15, 13], [13, 11], [11, 5], 
        [12, 14], [14, 16], [12, 6], 
        [3, 1],[1, 2],[1, 0],[0, 2],[2,4],
        [9, 7], [7,5], [5, 6],
        [6, 8], [8, 10],
        ]

point_color1 = [(240,2,127),(240,2,127),(240,2,127), 
            (240,2,127), (240,2,127), 
            (255,255,51),(255,255,51),
            (254,153,41),(44,127,184),
            (217,95,14),(0,0,255),
            (255,255,51),(255,255,51),(228,26,28),
            (49,163,84),(252,176,243),(0,176,240),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142)]

xiaochu_style = ColorStyle(color1, link_pairs1, point_color1)


# Chunhua Style
# (R,G,B)
color2 = [(252,176,243),(252,176,243),(252,176,243),
    (0,176,240), (0,176,240), (0,176,240),
    (240,2,127),(240,2,127),(240,2,127), (240,2,127), (240,2,127), 
    (255,255,0), (255,255,0),(169, 209, 142),
    (169, 209, 142),(169, 209, 142)]

link_pairs2 = [
        [15, 13], [13, 11], [11, 5], 
        [12, 14], [14, 16], [12, 6], 
        [3, 1],[1, 2],[1, 0],[0, 2],[2,4],
        [9, 7], [7,5], [5, 6], [6, 8], [8, 10],
        ]

point_color2 = [(240,2,127),(240,2,127),(240,2,127), 
            (240,2,127), (240,2,127), 
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (252,176,243),(0,176,240),(252,176,243),
            (0,176,240),(252,176,243),(0,176,240),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142)]

chunhua_style = ColorStyle(color2, link_pairs2, point_color2)

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize COCO predictions')
    # general
    parser.add_argument('--image-path',
                        help='Path of COCO val images',
                        type=str,
                        default='data/coco/images/val2017/'
                        )

    parser.add_argument('--gt-anno',
                        help='Path of COCO val annotation',
                        type=str,
                        default='data/coco/annotations/person_keypoints_val2017.json'
                        )

    parser.add_argument('--save-path',
                        help="Path to save the visualizations",
                        type=str,
                        default='visualization/coco/')

    parser.add_argument('--prediction',
                        help="Prediction file to visualize",
                        type=str,
                        required=True)

    parser.add_argument('--style',
                        help="Style of the visualization: Chunhua style or Xiaochu style",
                        type=str,
                        default='chunhua')

    args = parser.parse_args()

    return args


def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)
        
    return joints_dict

def plot(data, gt_file, img_path, save_path, 
         link_pairs, ring_color, save=True):
    
    # joints
    coco = COCO(gt_file)
    coco_dt = coco.loadRes(data)
    coco_eval = COCOeval(coco, coco_dt, 'keypoints')
    coco_eval._prepare()
    gts_ = coco_eval._gts
    dts_ = coco_eval._dts
    
    p = coco_eval.params
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)

    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]
    threshold = 0.3
    joint_thres = 0.2
    for catId in catIds:
        for imgId in p.imgIds[:5000]:
            # dimention here should be Nxm
            gts = gts_[imgId, catId]
            dts = dts_[imgId, catId]
            inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
            dts = [dts[i] for i in inds]
            if len(dts) > p.maxDets[-1]:
                dts = dts[0:p.maxDets[-1]]
            if len(gts) == 0 or len(dts) == 0:
                continue
            
            sum_score = 0
            num_box = 0
            img_name = str(imgId).zfill(12)
            
            # Read Images
            img_file = img_path + img_name + '.jpg'
            data_numpy = cv2.imread(img_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            h = data_numpy.shape[0]
            w = data_numpy.shape[1]
            
            # Plot
            fig = plt.figure(figsize=(w/100, h/100), dpi=100)
            ax = plt.subplot(1,1,1)
            bk = plt.imshow(data_numpy[:,:,::-1])
            bk.set_zorder(-1)
            print(img_name)
            for j, gt in enumerate(gts):
                # matching dt_box and gt_box
                bb = gt['bbox']
                x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
                y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2

                # create bounds for ignore regions(double the gt bbox)
                g = np.array(gt['keypoints'])
                #xg = g[0::3]; yg = g[1::3]; 
                vg = g[2::3]     
            
                for i, dt in enumerate(dts):
                    # Calculate IoU
                    dt_bb = dt['bbox']
                    dt_x0 = dt_bb[0] - dt_bb[2]; dt_x1 = dt_bb[0] + dt_bb[2] * 2
                    dt_y0 = dt_bb[1] - dt_bb[3]; dt_y1 = dt_bb[1] + dt_bb[3] * 2          
                    
                    ol_x = min(x1, dt_x1) - max(x0, dt_x0)
                    ol_y = min(y1, dt_y1) - max(y0, dt_y0)
                    ol_area = ol_x * ol_y
                    s_x = max(x1, dt_x1) - min(x0, dt_x0)
                    s_y = max(y1, dt_y1) - min(y0, dt_y0)
                    sum_area = s_x * s_y
                    iou = ol_area / (sum_area + np.spacing(1))                    
                    score = dt['score']
                    
                    if iou < 0.1 or score < threshold:
                        continue
                    else:
                        print('iou: ', iou)
                        dt_w = dt_x1 - dt_x0
                        dt_h = dt_y1 - dt_y0
                        ref = min(dt_w, dt_h)
                        num_box += 1
                        sum_score += dt['score']
                        dt_joints = np.array(dt['keypoints']).reshape(17,-1)
                        joints_dict = map_joint_dict(dt_joints)
                        
                        # stick 
                        for k, link_pair in enumerate(link_pairs):
                            if link_pair[0] in joints_dict \
                            and link_pair[1] in joints_dict:
                                if dt_joints[link_pair[0],2] < joint_thres \
                                    or dt_joints[link_pair[1],2] < joint_thres \
                                    or vg[link_pair[0]] == 0 \
                                    or vg[link_pair[1]] == 0:
                                    continue
                            if k in range(6,11):
                                lw = 1
                            else:
                                lw = ref / 100.
                            line = mlines.Line2D(
                                    np.array([joints_dict[link_pair[0]][0],
                                              joints_dict[link_pair[1]][0]]),
                                    np.array([joints_dict[link_pair[0]][1],
                                              joints_dict[link_pair[1]][1]]),
                                    ls='-', lw=lw, alpha=1, color=link_pair[2],)
                            line.set_zorder(0)
                            ax.add_line(line)
                        # black ring
                        for k in range(dt_joints.shape[0]):
                            if dt_joints[k,2] < joint_thres \
                                or vg[link_pair[0]] == 0 \
                                or vg[link_pair[1]] == 0:
                                continue
                            if dt_joints[k,0] > w or dt_joints[k,1] > h:
                                continue
                            if k in range(5):
                                radius = 1
                            else:
                                radius = ref / 100
                    
                            circle = mpatches.Circle(tuple(dt_joints[k,:2]), 
                                                     radius=radius, 
                                                     ec='black', 
                                                     fc=ring_color[k], 
                                                     alpha=1, 
                                                     linewidth=1)
                            circle.set_zorder(1)
                            ax.add_patch(circle)
        
            avg_score = (sum_score / (num_box+np.spacing(1)))*1000
        
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.axis('off')
            plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)        
            plt.margins(0,0)
            if save:
                plt.savefig(save_path + \
                           'score_'+str(np.int(avg_score))+ \
                           '_id_'+str(imgId)+ \
                           '_'+img_name + '.png', 
                           format='png', bbox_inckes='tight', dpi=100)
                plt.savefig(save_path +'id_'+str(imgId)+ '.pdf', format='pdf', 
                            bbox_inckes='tight', dpi=100)
            # plt.show()
            plt.close()

if __name__ == '__main__':

    args = parse_args()
    if args.style == 'xiaochu':
        # Xiaochu Style
        colorstyle = xiaochu_style
    elif args.style == 'chunhua':
        # Chunhua Style
        colorstyle = chunhua_style
    else:
        raise Exception('Invalid color style')
    
    save_path = args.save_path
    img_path = args.image_path
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except Exception:
            print('Fail to make {}'.format(save_path))

    
    with open(args.prediction) as f:
        data = json.load(f)
    gt_file = args.gt_anno
    plot(data, gt_file, img_path, save_path, colorstyle.link_pairs, colorstyle.ring_color, save=True)

