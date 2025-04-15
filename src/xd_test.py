import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import CLIPVAD
from utils.dataset import XDDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.xd_detectionMAP import getDetectionMAP as dmAP
import xd_option
from save_result import *

def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device, args):
    
    model.to(device)
    model.eval()

    element_logits2_stack = []
    video_labels_list = []
    video_names_list = []
    video_paths_list = []  # 프레임 이미지가 저장된 폴더 경로
    video_fps_list = []    # 동영상 fps (예: 30)

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            video_label = item[1] # add
            length = item[2] # padding 256 사이즈에서 실제 프레임 수만큼 줄이기
            cap_features = item[3].squeeze(0)
            cap_feat_lengths = item[4]
            video_basename = item[5] # add
            video_path = item[6]  # 프레임 이미지들이 저장된 폴더 경로
            video_fps = item[7]   # 동영상 fps

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
                cap_features = cap_features.unsqueeze(0)

            visual = visual.to(device)
            cap_features = cap_features.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            _, logits1, logits2, _, _ = model(visual, cap_features, padding_mask, prompt_text, lengths, cap_feat_lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            video_labels_list.append(video_label)
            video_names_list.append(video_basename)
            video_paths_list.append(video_path)
            video_fps_list.append(video_fps)

            if i == 0:
                ap1 = prob1
                ap2 = prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap1 = ap1.tolist()
    ap2 = ap2.tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    if args.saved_video:
        saved_test_video(args.gt_txt, video_names_list, element_logits2_stack, args.frame_base_folder, video_fps_list, prompt_text)

    print(f"AUC1: {ROC1:.4f}, AP1: {AP1:.4f}")
    print(f"AUC2: {ROC2:.4f}, AP2: {AP2:.4f}")

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP/(i+1)
    print('average MAP: {:.2f}'.format(averageMAP))

    if args.save_test_result:
        save_test_txt(ROC1, AP1, ROC2, AP2, averageMAP, dmap, iou, filename="../output/result_xd.txt")

    return ROC1, AP1, ROC2, AP2, averageMAP


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    test_dataset = XDDataset(args.visual_length, args.test_list, args.test_cap_list, True, label_map, using_caption=args.using_caption)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, args.batch_size, device)
    model_param = torch.load(args.model_path)
    # model_param = torch.load(args.checkpoint_path) # add
    # model_param = model_param['model_state_dict'] # add
    model.load_state_dict(model_param)

    test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device, args)