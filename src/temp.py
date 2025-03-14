import os
import csv

# npy 파일들이 저장된 루트 디렉토리
npy_root_dir = "/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/ucfclip_caption_feature(clean)"

# 입력 CSV 파일 경로 (각각 영상 이름과 레이블 정보를 포함한다고 가정)
train_input_csv = "/home/yeogeon/YG_main/diffusion_model/VadCLIP/list/ucf_CLIP_rgb_caption.csv"
test_input_csv = "/home/yeogeon/YG_main/diffusion_model/VadCLIP/list/ucf_CLIP_rgbtest_caption.csv"

# 출력 CSV 파일 경로
train_output_csv = "/home/yeogeon/YG_main/diffusion_model/VadCLIP/list/ucf_CLIP_rgb_caption(clean).csv"
test_output_csv = "/home/yeogeon/YG_main/diffusion_model/VadCLIP/list/ucf_CLIP_rgbtest_caption(clean).csv"

def process_csv(input_csv, output_csv):
    rows_to_write = []
    # CSV 파일 읽기 (첫 줄에 header가 있다고 가정)
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 헤더 스킵
        for row in reader:
            # 입력 CSV의 각 행은 영상 이름과 레이블을 포함 (예: Abuse001_x264, Abuse)
            if len(row) < 2:
                continue
            video = row[0].strip()
            video_basename = os.path.basename(video)
            label = row[1].strip()
            # npy 파일의 전체 경로 구성
            # 예: {npy_root_dir}/{label}/{video}/{video}.npy
            npy_path = os.path.join(npy_root_dir, f"{video_basename}")
            if os.path.exists(npy_path):
                rows_to_write.append([npy_path, label])
            else:
                print(f"파일을 찾을 수 없음: {npy_path}")
    # 출력 CSV 파일 저장 (header: path, label)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows_to_write)
    print(f"CSV 파일이 {output_csv}에 저장되었습니다.")

# train과 test CSV 각각 처리
process_csv(train_input_csv, train_output_csv)
process_csv(test_input_csv, test_output_csv)
