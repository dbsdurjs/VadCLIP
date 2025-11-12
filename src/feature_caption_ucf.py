# save as: cut_captions_first_sentence.py
import json
import re
import os, torch
from clip import clip
import numpy as np

# 간단한 문장 분리: 마침표/물음표/느낌표 뒤의 공백 기준
# _SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def load_clip_model():
    # CLIP 모델 로드 (ViT-B/32 사용, 필요에 따라 변경 가능)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/16", device=device)

    for clip_param in model.parameters():
        clip_param.requires_grad = False
    return model, device

def first_sentence(text: str | None = None) -> str:
    if not isinstance(text, str):
        text = str(text)

    # 줄바꿈/여러 공백을 단일 공백으로 정리
    text = text.split('.')[0].strip()
    if not text:
        return ""

    return text + "."

def process_json_to_clip_features(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    model, device = load_clip_model()

    for json_file in os.listdir(input_dir):
        if not json_file.endswith(".json"):
            continue
            
        json_path = os.path.join(input_dir, json_file)
        print(f"Processing {json_file}...")
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        captions = [first_sentence(v) for v in data.values() if v]
        if not captions:
            print(f"Skipping {json_file}: No valid captions")
            continue

        # 배치 크기 설정 (예: 32)
        batch_size = 32
        text_features_list = []
        
        for i in range(0, len(captions), batch_size):
            batch_captions = captions[i:i + batch_size]
            text_tokens = clip.tokenize(batch_captions).to(device)
            with torch.no_grad():
                batch_features = model.encode_text_cap(text_tokens).cpu().numpy()
            text_features_list.append(batch_features)
            torch.cuda.empty_cache()  # 배치 처리 후 메모리 정리
        
        # 모든 배치 결과를 하나로 합침
        text_features = np.concatenate(text_features_list, axis=0)

        output_filename = os.path.splitext(json_file)[0] + "_v2.npy"
        output_path = os.path.join(output_dir, output_filename)
        
        np.save(output_path, text_features)
        print(f"Processed {json_file}: {len(text_features)} items saved to {output_path}")

if __name__ == "__main__":

    input_dir = "../VAD_dataset/UCF-Crimes/Extracted_Captions_janus"
    output_dir = "../VAD_dataset/UCF-Crimes/Caption_Features_janus_v2"
    process_json_to_clip_features(input_dir, output_dir)
