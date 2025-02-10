import os
import csv

if __name__ == '__main__':
    # 비교 CSV 파일 경로 (클래스/동영상 이름이 기록된 파일)
    compare_csv_path = "/home/yeogeon/YG_main/diffusion_model/VadCLIP/list/ucf_CLIP_rgbtest_10videos.csv"
    
    # 비교 CSV 파일을 읽어서 유효한 (클래스, 동영상폴더 이름) 튜플의 집합을 구성합니다.
    valid_set = set()
    with open(compare_csv_path, 'r', encoding='utf-8') as f_compare:
        csv_reader = csv.reader(f_compare)
        header = next(csv_reader)  # 헤더가 있다면 읽어줍니다.
        for row in csv_reader:
            if len(row) < 2:
                continue
            old_path = row[0].strip()
            # 파일 경로에서 동영상 클래스를 추출합니다.
            path_parts = old_path.split(os.sep)
            if "UCFClipFeatures" in path_parts:
                video_class = path_parts[path_parts.index("UCFClipFeatures") + 1]
            else:
                video_class = "Unknown"  # 또는 다른 기본값
            # npy 파일의 basename 예: "Abuse002_x264__3.npy"
            base_name = os.path.basename(old_path)
            # "__"를 기준으로 분리하여 동영상 폴더 이름을 추출 (예: "Abuse002_x264")
            video_folder_name = base_name.split('__')[0]
            # valid_set에 (동영상 클래스, 동영상 폴더 이름) 튜플을 저장합니다.
            valid_set.add((video_class, video_folder_name))
    
    # 새로운 데이터셋의 최상위 폴더 (학습에 사용할 ucfclip_caption_feature 폴더)
    new_root_dir = '/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/ucfclip_caption_feature'
    # 새로 생성할 CSV 파일 경로
    new_csv_path = '/home/yeogeon/YG_main/diffusion_model/VadCLIP/list/ucf_CLIP_captiontest_10videos.csv'
    
    with open(new_csv_path, 'w', newline='', encoding='utf-8') as fout:
        csv_writer = csv.writer(fout)
        # 헤더 기록 (필요 시)
        csv_writer.writerow(["path", "label"])
        
        # 새로운 데이터셋 폴더 내의 각 클래스 폴더를 알파벳 순으로 정렬하여 순회
        for class_folder in sorted(os.listdir(new_root_dir)):
            class_path = os.path.join(new_root_dir, class_folder)
            if not os.path.isdir(class_path):
                continue
            
            # 각 클래스 폴더 내의 동영상 폴더도 정렬하여 순서대로 처리
            for video_folder in sorted(os.listdir(class_path)):
                video_folder_path = os.path.join(class_path, video_folder)
                if not os.path.isdir(video_folder_path):
                    continue
                
                # (클래스, 동영상폴더 이름) 튜플이 비교 CSV의 유효 집합에 존재하는지 확인
                if (class_folder, video_folder) in valid_set:
                    # 새 CSV에 기록할 경로는 동영상 폴더 이름에 ".npy" 확장자를 붙인 형태
                    new_file_path = os.path.join(new_root_dir, class_folder, video_folder, video_folder + ".npy")
                    csv_writer.writerow([new_file_path, class_folder])
                else:
                    print(f"Skipped: ({class_folder}, {video_folder}) not in compare CSV")
                    
    print(f"새로운 CSV 파일이 생성되었습니다: {new_csv_path}")
