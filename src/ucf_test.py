import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
import ucf_option
import os
import logging
from vis import *
import re
from save_result import *

# 로그 파일 설정
logging.basicConfig(
    filename='../output/test_results.log',  # 기록할 로그 파일 이름
    level=logging.INFO,           # INFO 레벨 이상의 로그를 기록
    format='%(asctime)s %(levelname)s: %(message)s',  # 로그 메시지 포맷
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2]) # (batch, 256, 1) -> (256, 1)
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2]) # (batch, 256, 14) -> (256, 14)

            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1)) # normal 클래스 값들 추출해서 1에서 빼줌
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            video_labels_list.append(video_label)
            video_names_list.append(video_basename)
            video_paths_list.append(video_path)
            video_fps_list.append(video_fps)

            if i == 0:
                ap1 = prob1
                ap2 = prob2
                #ap3 = prob3
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy() # (clip 수, 14)
            element_logits2 = np.repeat(element_logits2, 16, 0) # 16으로 나누기 전으로 돌리는? (clip 수 * 16, 클래스 수), 아마 인접 16프레임은 anomaly가 일어나지 않기에 이렇게 계산하는듯?
            element_logits2_stack.append(element_logits2)

            # 비디오 전체에 대한 평균 anomaly score 계산
            anomaly_score_ap1 = prob1.mean().item() # add
            anomaly_score_ap2 = prob2.mean().item() # add
            
            # 로그 파일에 결과 기록
            logging.info(f"Video: {video_basename}, GT Label: {video_label}, "
                        f"Predicted anomaly score (AP1): {anomaly_score_ap1:.4f}, "
                        f"AP2: {anomaly_score_ap2:.4f}") # add

    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap1 = ap1.tolist()
    ap2 = ap2.tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))    # sigmoid 방식(binary classification)
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))   # sigmoid 방식(binary classification)

    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))    # softmax 방식(multi-class classification)
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))   # softmax 방식(multi-class classification)
    
    if args.saved_video:
        saved_test_video(args.gt_txt, video_names_list, element_logits2_stack, args.frame_base_folder, video_fps_list, prompt_text)

    # gt는 동영상 프레임에 대한 n/a를 나타냄 [0,0,0,1,1,...]
    # ROC1 : C-branch에서 직접 anomaly confidence를 구하는 법
    # ROC2 : A-branch에서 간접 anomaly confidence를 구하는 법(1-normal class 확률 = anomaly class 확률)
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
        save_test_txt(ROC1, AP1, ROC2, AP2, averageMAP, dmap, iou, filename="../output/result_ucf.txt")

    return ROC1, AP1, ROC2, AP2, averageMAP


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()

    label_map = dict({'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion', 'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery', 'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'})

    testdataset = UCFDataset(args.visual_length, args.test_list, args.test_cap_list, True, label_map, using_caption=args.using_caption)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)
    #test 시 동영상 1개씩 처리
    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, args.batch_size, device)
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param)

    test(model, testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device, args)