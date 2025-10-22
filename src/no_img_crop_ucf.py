import os
import re
import torch
import numpy as np
from clip import clip
from PIL import Image
from tqdm import tqdm

def extract_clip_features_from_images(image_dir, output_path, model, preprocess, device):
    """
    이미지 폴더 내 모든 .jpg 파일에 대해 CLIP feature 추출 후 npy로 저장
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
    frame_features = []

    with torch.no_grad():
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"⚠️ {img_file} 로드 실패: {e}")
                continue

            image_input = preprocess(image).unsqueeze(0).to(device)
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            frame_features.append(image_features.cpu().numpy())

    if len(frame_features) == 0:
        print(f"❌ {image_dir} 내에 유효한 이미지가 없습니다.")
        return

    frame_features = np.concatenate(frame_features, axis=0)
    np.save(output_path, frame_features)
    print(f"✅ {image_dir}: {len(frame_features)}장의 이미지 feature 저장 완료 → {output_path}")

if __name__ == '__main__':
    # 동영상 기본 경로
    base_path = '../VAD_dataset/UCF-Crimes/Extracted_Frames'
    # feature 저장 경로
    save_base = '../VAD_dataset/UCFClipFeatures_NoCrop'
    os.makedirs(save_base, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    for folder_name in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # ✅ 정규식만으로 class 이름 추출: 'Abuse001_x264' → 'abuse'
        class_name = re.sub(r'[\d_].*$', '', folder_name)

        save_dir = os.path.join(save_base, class_name)
        os.makedirs(save_dir, exist_ok=True)

        output_path = os.path.join(save_dir, f"{folder_name}.npy")
        extract_clip_features_from_images(folder_path, output_path, model, preprocess, device)