import os
import csv
#10개 비디오에 대한 csv 파일 만들기
#기존 ucf_CLIP_rgb.csv로 추출한 10개 비디오의 경로만 받아서 ucf_CLIP_rgb_10videos.csv 생성

if __name__ == '__main__':

    # 기존 CSV 파일 경로 (예: UCFClipFeatures에 해당하는 CSV)
    old_csv_path = "/home/yeogeon/YG_main/diffusion_model/VadCLIP/list/ucf_CLIP_rgb.csv"  # train
    # old_csv_path = "/home/yeogeon/YG_main/diffusion_model/VadCLIP/list/ucf_CLIP_rgbtest.csv"    # test

    # 새로운 데이터셋의 최상위 root 디렉토리 (학습에 사용할 ucfclip_caption_feature 폴더)
    new_root_dir = '/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Extracted_Frames_with_10videos'
    # 생성할 새로운 CSV 파일 경로
    new_csv_path = "/home/yeogeon/YG_main/diffusion_model/VadCLIP/list/ucf_CLIP_rgb_caption.csv" # train
    # new_csv_path = "/home/yeogeon/YG_main/diffusion_model/VadCLIP/list/ucf_CLIP_rgbtest_10videos.csv"   # test

    with open(old_csv_path, 'r', encoding='utf-8') as fin, \
         open(new_csv_path, 'w', newline='', encoding='utf-8') as fout:
        csv_reader = csv.reader(fin)
        csv_writer = csv.writer(fout)
        
        # 헤더가 있다면 기록
        header = next(csv_reader)
        csv_writer.writerow(header)
        
        for row in csv_reader:
            # CSV 파일의 한 행은 [npy파일경로, 기존_라벨] 형식입니다.
            if len(row) < 2:
                continue  # 형식이 맞지 않는 경우 넘어감
            
            old_path = row[0].strip()
            # CSV의 두 번째 열은 무시하고, npy 파일 경로에서 label을 추출합니다.
            path_parts = old_path.split(os.sep)
            if "UCFClipFeatures" in path_parts:
                label_from_path = path_parts[path_parts.index("UCFClipFeatures") + 1]
            else:
                # 만약 "UCFClipFeatures"가 없으면 기존 CSV의 label을 사용합니다.
                label_from_path = row[1].strip()
            
            # npy 파일의 basename 추출
            base_name = os.path.basename(old_path)  # 예: "Abuse002_x264__3.npy"
            # "__"를 기준으로 분리하여, 첫 번째 부분을 동영상 폴더 이름으로 사용
            video_folder_name = base_name.split('__')[0]  # 예: "Abuse002_x264"
            
            # 새로운 데이터셋 내에서 해당 동영상 폴더의 경로 구성
            new_video_folder = os.path.join(new_root_dir, label_from_path, video_folder_name)
            
            # 새로운 데이터셋에 해당 동영상 폴더가 존재하면 CSV에 행 기록
            if os.path.isdir(new_video_folder):
                csv_writer.writerow(row)
            else:
                print(f"폴더 없음: {new_video_folder}")

    print(f"새로운 CSV 파일이 생성되었습니다: {new_csv_path}")
