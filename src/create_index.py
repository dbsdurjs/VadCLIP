# 비디오 캡션을 로드하여, CLIP(텍스트 인코더)로 임베딩한 뒤, FAISS 인덱스를 생성 및 저장
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import faiss
import torch
from tqdm import tqdm
import re
from clip import clip

from clip_model_init import initialize_vlm_model_and_device
from image_text_caption_cleaner import *

base_path = Path('/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Extracted_Frames/')

def cal_total_frames(output_caption_json):
    with open(output_caption_json) as f:
        video_data = json.load(f)

    if video_data:
        frame_indices = [int(k) for k in video_data.keys()]  # "00000" -> 0
        min_idx = min(frame_indices)
        max_idx = max(frame_indices)
        total_frames = max_idx - min_idx + 1
        print(f"[DEBUG] frame range: {min_idx} ~ {max_idx}, total_frames={total_frames}")
    
    return video_data, total_frames

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--frame_interval", type=int, default=16)
    return parser.parse_args()

def process_video(
    video,
    model: torch.nn.Module,
    device: str,
    index_dim: int,
    batch_size: int,
    frame_interval: int,
    captions_dirs: List[str],
    output_dir: Path,
):
    video_name = video.name # ex)Abuse005_x264.mp4 / stem ex) Abuse005_x264
    index = init_faiss_index(index_dim)
    file_names = []
    video_captions, total_frames = load_video_captions(captions_dirs, video_name)

    caption_to_frame_idxs = build_caption_to_frame_index(video_captions) # 같은 캡션에 대한 프레임 인덱스 값을 리스트로 저장
    # {'a man is standing in a room with a computer': [0, 1, 2, 3, ..]
    for batch_start_frame in tqdm(
        range(0, total_frames, batch_size * frame_interval),
        desc=f"Processing {video_name}",
        unit="batch",
    ):
        batch_end_frame = min(batch_start_frame + (batch_size * frame_interval), total_frames)
        batch_frame_idxs = range(batch_start_frame, batch_end_frame, frame_interval)

        text_list = extract_text_list(
            video_captions, caption_to_frame_idxs, batch_frame_idxs, frame_interval
        )

        if text_list:
            index = update_faiss_index(model, device, index, text_list)
            file_names.extend(
                build_file_names(
                    video_captions, caption_to_frame_idxs, text_list, video_name, frame_interval
                )
            )

    save_results(index, file_names, output_dir, video_name)


def init_faiss_index(index_dim: int):
    index = faiss.IndexFlatIP(index_dim) # cosine sim 구하기
    return index

def parse_frame_caption_file(input_txt_path, output_json_path=None):
    # txt -> json
    video_captions = {}
    pattern = re.compile(r'_frame_(\d+)\.jpg$')
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # ': ' 기준으로 나누되, 최대 1회만 split (캡션 중에 ': '가 더 있을 수도 있으므로)
            if ': ' in line:
                frame_name, caption = line.split(': ', 1)
                match = pattern.search(frame_name)
                if match:
                    frame_idx = match.group(1)  # "00001"
                    video_captions[frame_idx] = caption
            else:
                # 혹시 ': '가 없는 라인은 예외처리
                print(f"[Warning] 구분자(': ')가 없는 라인입니다: {line}")

    # JSON 저장(옵션)
    if output_json_path:
        with open(output_json_path, 'w', encoding='utf-8') as out_f:
            json.dump(video_captions, out_f, indent=4, ensure_ascii=False)

    return video_captions

def load_video_captions(captions_dirs: List[str], video_names: str):
    video_captions = defaultdict(dict)
    video_name = video_names.split('.')[0]

    match = re.match(r'([A-Za-z]+)', video_name)
    if match:
        class_prefix = match.group(1)  # "Abuse"
        if class_prefix == 'Normal':
            class_prefix = 'Training_Normal_Videos_Anomaly'

    captions_dir = Path(captions_dirs)

    video_caption_txt = captions_dir / class_prefix / video_name / f"{video_name}.txt"
    output_caption_json = captions_dir / class_prefix / video_name / f"{video_name}.json"
    parse_frame_caption_file(video_caption_txt, output_json_path=output_caption_json)
    
    video_data, total_frames = cal_total_frames(output_caption_json)

    for frame_idx, caption in video_data.items():
        video_captions[frame_idx] = caption

    return video_captions, total_frames

def build_caption_to_frame_index(video_captions):
    caption_to_frame_idxs = defaultdict(list)
    for frame_idx, cap_model_name_to_caption in video_captions.items():
        frame_unique_captions = [cap_model_name_to_caption]
        for caption in frame_unique_captions:
            caption_to_frame_idxs[caption].append(int(frame_idx))

    return caption_to_frame_idxs

def extract_text_list(video_captions, caption_to_frame_idxs, batch_frame_idxs, frame_interval):
    text_list = []
    for frame_idx in batch_frame_idxs:
        str_idx = f"{frame_idx:05d}"

        frame_unique_captions = [video_captions[str(str_idx)]]
        unique_captions = [
            caption
            for caption in frame_unique_captions
            if int(str_idx)
            == min(filter(lambda x: x % frame_interval == 0, caption_to_frame_idxs[caption]))
        ]
        text_list.extend(unique_captions)
    return text_list

def update_faiss_index(model, device, index, text_list):
    inputs = text_list
    with torch.no_grad():
        tokens = clip.tokenize(inputs).to(device)
        embeddings = model.encode_text_cap(tokens)
        embeddings = embeddings.float().detach().cpu().numpy()
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

    return index


def build_file_names(video_captions, caption_to_frame_idxs, text_list, video_name, frame_interval):
    file_names = []
    for caption in text_list:
        frame_idx = min(filter(lambda x: x % frame_interval == 0, caption_to_frame_idxs[caption])) # 각 캡션에 해당하는 프레임 중 16의 배수의 최소 인덱스 프레임
        file_names.append(f"{video_name}/{frame_idx}")

    return file_names


def save_results(index, file_names, output_dir, video_name):
    # Save faiss index
    faiss.write_index(index, str(output_dir / f"{video_name}.bin"))
    # Save file names
    with open(output_dir / f"{video_name}.json", "w") as f:
        json.dump(file_names, f)


def main(
    index_dim: int,
    batch_size: int,
    frame_interval: int,
    captions_dirs: List[str],
    output_dir: str,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _, device = initialize_vlm_model_and_device()
    video_list = load_video()

    for video in video_list:
        process_video(
            video,
            model,
            device,
            index_dim,
            batch_size,
            frame_interval,
            captions_dirs,
            output_dir,
        )


if __name__ == "__main__":
    args = parse_args()
    captions_dirs = '/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Extracted_Frames_captions'
    output_dir = '/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/create_index'
    main(
        args.index_dim, # 1024
        args.batch_size, # 64
        args.frame_interval, # 16
        captions_dirs=captions_dirs,
        output_dir=output_dir
    )
