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

# 로그 파일 설정
logging.basicConfig(
    filename='test_results.log',  # 기록할 로그 파일 이름
    level=logging.INFO,           # INFO 레벨 이상의 로그를 기록
    format='%(asctime)s %(levelname)s: %(message)s',  # 로그 메시지 포맷
    datefmt='%Y-%m-%d %H:%M:%S'
)

frame_base_folder = "/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Extracted_Frames/"
gt_txt = './list/Temporal_Anomaly_Annotation.txt'

def find_video_folder(vname):
    for root, dirs, files in os.walk(frame_base_folder):
        for d in dirs:
            if d == vname:
                return os.path.join(root, d)
    return None

def read_annotation_intervals(annotation_file):
    """
    annotation_file: 각 줄이 "videoName  Class  start1  end1  start2  end2 ..." 형식인 텍스트 파일 경로.
    반환: { video_name: [(start1, end1), (start2, end2), ...] }
           -1인 값은 무시합니다.
    """
    annotations = {}
    with open(annotation_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 4:
                continue  # 최소한 videoName, Class, start, end가 있어야 함.
            video_name = tokens[0]  
            intervals = []
            # 두 번째 토큰은 클래스 정보이므로, 2번 인덱스부터 시작
            for i in range(2, len(tokens), 2):
                start = int(tokens[i])
                end = int(tokens[i+1]) if i+1 < len(tokens) else -1
                if start != -1 and end != -1:
                    intervals.append((start, end))
            annotations[video_name] = intervals
    return annotations

exclude_list = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'RoadAccidents', 'Robbery']
def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device, saved_video):
    
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
            _, logits1, logits2 = model(visual, cap_features, padding_mask, prompt_text, lengths, cap_feat_lengths)

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
    
    if saved_video:
        anno = read_annotation_intervals(gt_txt)
        for idx in range(len(video_names_list)):
            vname = video_names_list[idx][0].split('__')[0] if isinstance(video_names_list[idx], tuple) else video_names_list[idx]
            
            # match = re.match(r'([A-Za-z]+)', vname)   # 미리 했던 작업이 있어서 제외하고 나머지 작업 코드
            # class_prefix = match.group(1) if match else vname

            # if class_prefix in exclude_list:
            #     continue
            frame_path = find_video_folder(vname)

            avg_class_scores = np.mean(element_logits2_stack[idx], axis=0)
            best_class_idx = np.argmax(avg_class_scores)
            vscores = element_logits2_stack[idx][:, best_class_idx]  # shape: (프레임 수,)

            vpath = os.path.join(frame_base_folder, frame_path)
            vfps = video_fps_list[idx][0]        # 동영상 fps
            
            # 저장 경로 지정 (예: vname.mp4 파일)
            save_path = os.path.join(vpath, f"{vname}_visualization.mp4")
            normal_label = prompt_text[0]  # 예시
            imagefile_template = vname + "_frame_{:05d}.jpg" 
            
            anno_for_video = anno.get(vname, [])
            visualize_video(vname, anno_for_video, vscores, vpath, vfps, save_path, normal_label, imagefile_template, None)
            logging.info(f"Visualization video saved: {save_path}")
            print(f"Visualization video saved: {save_path}")

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

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param)

    test(model, testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device, args.saved_video)