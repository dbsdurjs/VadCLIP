import datetime
from vis import *
import os
import numpy as np

def save_test_txt(ROC1, AP1, ROC2, AP2, averageMAP, dmap, iou, filename):
    result_filename = filename
    with open(result_filename, "a", encoding="utf-8") as file:
        # 현재 날짜 및 시간 가져오기
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # 지정한 형식으로 파일에 작성
        file.write("----------------------------------------------------------------\n")
        file.write(f"저장 날짜 : {date_str}\n")
        file.write(f"AUC1 : {ROC1:.4f}, AP1: {AP1:.4f}\n")
        file.write(f"AUC2 : {ROC2:.4f}, AP2: {AP2:.4f}\n")
        for i in range(len(dmap)):
            file.write('mAP@{:.1f} = {:.2f}%\n'.format(iou[i], dmap[i]))
        file.write("average MAP: {:.2f}\n".format(averageMAP))
        file.write("-----------------------------------------------------------------\n")

    print(f"결과가 {result_filename} 파일에 저장되었습니다.")

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

def find_video_folder(vname, frame_base_folder):
    for root, dirs, files in os.walk(frame_base_folder):
        for d in dirs:
            if d == vname:
                return os.path.join(root, d)
    return None

def saved_test_video(gt_txt, video_names_list, element_logits2_stack, frame_base_folder, video_fps_list, prompt_text):
        anno = read_annotation_intervals(gt_txt)
        for idx in range(len(video_names_list)):
            vname = video_names_list[idx][0].split('__')[0] if isinstance(video_names_list[idx], tuple) else video_names_list[idx]
            
            # match = re.match(r'([A-Za-z]+)', vname)   # 미리 했던 작업이 있어서 제외하고 나머지 작업 코드
            # class_prefix = match.group(1) if match else vname

            # if class_prefix in exclude_list:
            #     continue
            frame_path = find_video_folder(vname, frame_base_folder)

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
            print(f"Visualization video saved: {save_path}")