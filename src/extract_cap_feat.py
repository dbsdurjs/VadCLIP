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
    root_text_dir = "/media/vcl/DATA/YG/Extracted_Frames"
    output_dir = "/media/vcl/DATA/YG/VAD_dataset/UCF-Crimes/UCF_Crimes/ucfclip_caption_feature"
    os.makedirs(output_dir, exist_ok=True)

    group_size = 16
    batch_size = 32  # 한 번에 처리할 캡션 개수를 줄임

    # 재귀적으로 텍스트 파일 찾기
    for dirpath, dirnames, filenames in os.walk(root_text_dir):
        for filename in filenames:
            if filename.endswith(".txt"):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                captions = []
                for line in lines:
                    parts = line.strip().split(":", 1)
                    if len(parts) == 2:
                        caption = parts[1].strip()
                        captions.append(caption)
                if len(captions) == 0:
                    print(f"캡션이 없는 파일: {file_path}")
                    continue

                # 배치 처리
                all_features = []
                for i in range(0, len(captions), batch_size):
                    batch_captions = captions[i:i+batch_size]
                    tokens = clip.tokenize(batch_captions).to(device)
                    # 여기서는 model.encode_text_cap()를 사용하지만, 표준 CLIP이라면 model.encode_text(tokens)를 사용
                    batch_features = model.encode_text_cap(tokens)
                    batch_features = batch_features.float().detach().cpu().numpy()
                    all_features.append(batch_features)
                    
                    # 배치 처리 후 캐시 비우기
                    torch.cuda.empty_cache()
                text_features = np.concatenate(all_features, axis=0)  # shape: [전체 캡션 수, feature_dim]

                grouped_features = average_features(text_features, group_size=16)
                rel_path = os.path.relpath(dirpath, root_text_dir)
                out_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(out_subdir, exist_ok=True)
                output_file = os.path.join(out_subdir, filename.replace(".txt", ".npy"))
                np.save(output_file, grouped_features)
                print(f"Saved {output_file} with shape {grouped_features.shape}")