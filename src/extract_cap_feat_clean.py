# 기존 caption feature, visual feature 들은 16으로 나누어떨어지지 않는 맨 마지막을 버림
# clean caption.json은 각 대표 프레임을 나타내기에 
import os
import torch
from clip import clip
import numpy as np
from compare import average_features
import json
from clip_model_init import initialize_vlm_model_and_device

if __name__ == '__main__':
    model, _, device = initialize_vlm_model_and_device()

    root_text_dir = "/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/captions/clean"
    output_dir = "/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/ucfclip_caption_feature(clean)"
    os.makedirs(output_dir, exist_ok=True)

    # root_text_dir 내의 모든 JSON 파일 재귀적으로 탐색
    for dirpath, dirnames, filenames in os.walk(root_text_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                json_path = os.path.join(dirpath, filename)
                print(f"파일 처리 중: {filename}")
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                all_features = []
                # 각 JSON 파일의 외부 키(대표 프레임)에 대해 처리
                for rep_frame, captions_dict in data.items():
                    # 내부 딕셔너리의 키(프레임 번호)를 기준으로 정렬하여 10개의 캡션 순서 보장
                    sorted_items = sorted(captions_dict.items(), key=lambda x: int(x[0]))
                    captions = [caption for _, caption in sorted_items]

                    # 캡션 10개에 대해 토큰화 및 CLIP 텍스트 임베딩 추출
                    tokens = clip.tokenize(captions).to(device)
                    with torch.no_grad():
                        features = model.encode_text_cap(tokens)
                    features = features.float().detach().cpu().numpy()

                    # 10개의 캡션 임베딩을 단순 평균으로 하나의 벡터로 집계
                    avg_group = average_features(features, group_size=10)
                    all_features.append(avg_group)

                if len(all_features) // 16 != 0:
                    all_features = all_features[:-1]
                    
                text_features = np.concatenate(all_features, axis=0)
                # 결과 파일 저장을 위해 원본 경로에 맞는 상대 경로 생성
                rel_path = os.path.relpath(dirpath, root_text_dir)
                out_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(out_subdir, exist_ok=True)

                output_file = os.path.join(out_subdir, filename.replace(".json", ".npy"))
                np.save(output_file, text_features)
                print(f"파일 이름 : {filename} | 임베딩 결과 : {text_features.shape}")
