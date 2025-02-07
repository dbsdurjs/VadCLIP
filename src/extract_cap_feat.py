import os
import torch
from clip import clip
import numpy as np

if __name__ == '__main__':
    # device 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CLIP 모델 로드 (ViT-B/16 사용)
    model, _ = clip.load("ViT-B/16", device)
    for clip_param in model.parameters():
        clip_param.requires_grad = False
    model.eval()

    # 텍스트 파일들이 있는 폴더와 결과 npy 파일을 저장할 폴더 설정
    text_files_dir = "/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Extracted_Frames/Fighting/Fighting003_x264"
    output_dir = "/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/ucfclip_caption_feature"
    os.makedirs(output_dir, exist_ok=True)

    group_size = 16  # 16 프레임(캡션)씩 그룹화

    # 확장자가 .txt 인 파일 목록 가져오기
    text_files = [f for f in os.listdir(text_files_dir) if f.endswith(".txt")]

    with torch.no_grad():
        for text_file in text_files:
            file_path = os.path.join(text_files_dir, text_file)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # 각 라인에서 콜론(:) 이후의 캡션 텍스트만 추출
            captions = []
            for line in lines:
                parts = line.strip().split(":", 1)
                if len(parts) == 2:
                    caption = parts[1].strip()
                    captions.append(caption)

            if len(captions) == 0:
                print(f"캡션이 없는 파일: {text_file}")
                continue

            # CLIP 텍스트 인코더로 캡션 feature 추출
            tokens = clip.tokenize(captions).to(device)
            # CLIP encode_text_cap 코드 만듦
            text_features = model.encode_text_cap(tokens)   # shape: [num_captions, feature_dim]
            text_features = text_features.float().detach().cpu().numpy()

            # 16캡션씩 그룹화하여 각 그룹의 평균 feature 계산
            num_captions = text_features.shape[0]
            num_groups = num_captions // group_size  # 나머지는 버림
            if num_groups == 0:
                print(f"{text_file} 파일은 그룹화할 만큼 충분한 캡션이 없습니다.")
                continue

            grouped_features = []
            for i in range(num_groups):
                group = text_features[i * group_size:(i + 1) * group_size]
                group_avg = np.mean(group, axis=0)   # 그룹 내 캡션들의 평균 feature
                grouped_features.append(group_avg)
            grouped_features = np.stack(grouped_features, axis=0)  # shape: (num_groups, feature_dim)

            # npy 파일로 저장 (예: Fighting003_x264.txt -> Fighting003_x264.npy)
            output_file = os.path.join(output_dir, text_file.replace(".txt", ".npy"))
            np.save(output_file, grouped_features)
            print(f"Saved {output_file} with shape {grouped_features.shape}")
