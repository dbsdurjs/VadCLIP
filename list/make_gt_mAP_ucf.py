import numpy as np
import glob
import os
import cv2
import pandas as pd
import warnings

clip_len = 16

feature_list = 'list/ucf_CLIP_rgbtest_10videos.csv'

# the ground truth txt
gt_txt = 'list/Temporal_Anomaly_Annotation.txt'
gt_lines = list(open(gt_txt))

#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

gt_segment = []
gt_label = []
lists = pd.read_csv(feature_list)

for idx in range(lists.shape[0]):
    name = lists.loc[idx]['path']
    label_text = lists.loc[idx]['label']
    if '__0.npy' not in name:
        continue
    segment = []
    label = []
    if 'Normal' in label_text:
        fea = np.load(name)
        lens = fea.shape[0] * clip_len
        name = name.split('/')[-1]
        name = name[:-7]
        segment.append([0, lens])
        label.append('A')
    else:
        name = name.split('/')[-1]
        name = name[:-7]
        for gt_line in gt_lines:
            if name in gt_line:
                gt_content = gt_line.strip('\n').split('  ')
                segment.append([gt_content[2], gt_content[3]])
                label.append(gt_content[1])
                if gt_content[4] != '-1':
                    segment.append([gt_content[4], gt_content[5]])
                    label.append(gt_content[1])
                break
    gt_segment.append(segment)
    gt_label.append(label)

# np.save('list/gt_label_ucf.npy', gt_label, allow_pickle=True)
# np.save('list/gt_segment_ucf.npy', gt_segment, allow_pickle=True)
np.save('list/gt_label_ucf_10videos.npy', np.array(gt_label, dtype=object))
np.save('list/gt_segment_ucf_10videos.npy', np.array(gt_segment, dtype=object))
