# 논문에서는 모든 프레임에 대해서 캡션을 비교 후 가장 유사한 캡션으로 교체하는 작업 진행
# 코드에서는 제한 범위 내에 가장 유사한 캡션을 뽑음
# num_samples 수만큼 서브 프레임을 뽑아서 각각 1개의 유사 캡션을 뽑은 후 대표 프레임은 총 10개의 캡션을 가지게 됨
# 실험 해보기(기존 vs 대표 프레임이 1개의 캡션을 가지기)
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
import os
import PIL.Image
import faiss 
import numpy as np
import torch
from tqdm import tqdm
import PIL
import pandas as pd
# from src.utils.path_utils import find_unprocessed_videos
from sample_util import uniform_temporal_subsample
from clip_model_init import initialize_vlm_model_and_device
from create_index import *

def load_video_ucf(root_path, filename: str):
    video_list = []
    with open(filename, 'r', encoding='utf-8') as f:   
        for line in f:
            relatvite_path = Path(line.strip().split('.')[0]) # Abuse/Abuse002_x264
            if relatvite_path:
                full_path = root_path / relatvite_path # ../VAD_dataset/UCF-Crimes/Extracted_Frames/Abuse/Abuse002_x264
                video_list.append(full_path)
    
    return video_list

def load_video_xd(root_path, filename: str):
    # CSV 파일 읽기
    df = pd.read_csv(filename)

    # 중복을 피하기 위해 set 사용
    video_lists = set()

    # 각 경로에서 파일명 추출 후, 마지막 "__"를 기준으로 분리하여 동영상 이름만 얻음
    for path in df['path']:
        base_name = os.path.basename(path)        # 예: Bad.Boys.1995__#00-26-51_00-27-53_label_B2-0-0__0.npy
        video_name = base_name.rsplit("__", 1)[0]    # 예: Bad.Boys.1995__#00-26-51_00-27-53_label_B2-0-0
        video_list = os.path.join(root_path, video_name, video_name)
        video_lists.add(video_list)

    # 결과를 리스트로 변환
    video_list = list(video_lists)
    return video_list


class ImageTextCaptionCleaner:
    def __init__(
        self,
        model,
        device,
        output_dir,
        num_samples,
        num_neighbors,
        index_dir,
        captions_dir_template,
        clip_duration,
        fps,
        imagefile_template,
        batch_size,
        frame_interval,
        preprocess,
        dataset
    ):
        self.model = model
        self.device = device
        self.preprocess = preprocess
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples # 10
        self.num_neighbors = num_neighbors # 1
        self.index_dir = index_dir 
        self.captions_dir_template = captions_dir_template
        self.clip_duration = clip_duration # 10
        self.fps = fps # 30
        self.imagefile_template = imagefile_template
        self.batch_size = batch_size # 64
        self.frame_interval = frame_interval # 16

    def process_video(self, video, dataset):
        video_name = Path(video).name # /home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Extracted_Frames/class/video name
        frames_per_clip = int(self.clip_duration * self.fps) # 30 * 10
        video_captions_retrieved = defaultdict(dict)

        index = self._load_faiss_index(video_name)
        file_names = self._load_file_names(video_name)

        if dataset == 'UCF-Crimes':
            match = re.match(r'([A-Za-z]+)', video_name)
            if match:
                class_prefix = match.group(1)  # "Abuse"
                if class_prefix == 'Normal':
                    class_prefix  = 'Testing_Normal_Videos_Anomaly' # Train or Test       

            frames_json_path = self.captions_dir_template / Path(class_prefix) / video_name / f'{video_name}.json'
        
        elif dataset == 'XD-Violence':
            frames_json_path = self.captions_dir_template / f'{video_name}.json'

        _, total_frames = cal_total_frames(frames_json_path)

        for batch_start_frame in tqdm(
            range(0, total_frames, self.batch_size * self.frame_interval),  # frame_interval 16
            desc=f"Processing {video_name}",
            unit="batch",
        ):
            batch_end_frame = min(
                batch_start_frame + (self.batch_size * self.frame_interval), total_frames
            )
            batch_center_frame_idxs = range(
                batch_start_frame, batch_end_frame, self.frame_interval
            )
            batch_frame_idxs, batch_frame_paths = self._prepare_frame_data(
                video, batch_center_frame_idxs, frames_per_clip, total_frames, video_name, dataset
            )

            inputs = self._load_and_transform_data(batch_frame_paths)
            search_vectors = self._calculate_search_vectors(inputs).float().detach().cpu().numpy()
            faiss.normalize_L2(search_vectors)

            distances, indices = index.search(search_vectors, self.num_neighbors)
            self._retrieve_captions(
                indices,
                batch_frame_idxs,
                file_names,
                batch_center_frame_idxs,
                video_captions_retrieved,
                dataset
            )

        self._save_results(video_name, video_captions_retrieved)

    def _load_faiss_index(self, video_name):
        index_file_path = Path(self.index_dir) / f"{video_name}.bin"
        return faiss.read_index(str(index_file_path))

    def _load_file_names(self, video_name):
        file_names_file_path = Path(self.index_dir) / f"{video_name}.json"
        with open(file_names_file_path) as f:
            return json.load(f)

    def batch_clip_frame_paths_xd(self, video, batch_center_frame_idxs, frames_per_clip, total_frames):
        batch_clip_frame_paths = [
            [
                self.imagefile_template.format(video, frame_idx)
                for frame_idx in range(
                    max(clip_center_frame - frames_per_clip // 2, 0),
                    min(clip_center_frame + frames_per_clip // 2, total_frames),
                )
            ]
            for clip_center_frame in batch_center_frame_idxs
        ]

        return batch_clip_frame_paths
    
    def batch_clip_frame_paths_ucf(self, video, batch_center_frame_idxs, frames_per_clip, total_frames, video_name):
        batch_clip_frame_paths = [
            [
                Path(video) / self.imagefile_template.format(video_name, frame_idx)
                for frame_idx in range(
                    max(clip_center_frame - frames_per_clip // 2, 0),
                    min(clip_center_frame + frames_per_clip // 2, total_frames),
                )
            ]
            for clip_center_frame in batch_center_frame_idxs
        ]

        return batch_clip_frame_paths

    def _prepare_frame_data(self, video, batch_center_frame_idxs, frames_per_clip, total_frames, video_name, dataset):
        if dataset == 'UCF-Crimes':
            batch_clip_frame_paths = self.batch_clip_frame_paths_ucf(video, batch_center_frame_idxs, frames_per_clip, total_frames, video_name)
        elif dataset == 'XD-Violence':
            batch_clip_frame_paths = self.batch_clip_frame_paths_xd(video, batch_center_frame_idxs, frames_per_clip, total_frames)
        
        batch_clip_subsample_frame_paths = [
            uniform_temporal_subsample(clip_frame_paths, self.num_samples)
            for clip_frame_paths in batch_clip_frame_paths
        ] # num sample 수만큼 균등하게 샘플링

        batch_frame_paths = [
            frame_path
            for clip_frame_paths in batch_clip_subsample_frame_paths
            for frame_path in clip_frame_paths
        ]

        # filenames' name is the frame index
        batch_frame_idxs = [int(Path(frame_path).stem.split('_')[-1]) for frame_path in batch_frame_paths]
        return batch_frame_idxs, batch_frame_paths

    def _load_and_transform_data(self, batch_frame_paths):
        images = []
        for img in batch_frame_paths:
            img = self.preprocess(PIL.Image.open(img).convert('RGB')).to(self.device)
            images.append(img)
        inputs = torch.stack(images, dim=0).to(self.device)
        return inputs

    def _calculate_search_vectors(self, inputs): # embedding vector 얻기
        with torch.no_grad():
            embeddings = self.model.encode_image(inputs)
        return embeddings

    def _retrieve_captions(
        self,
        indices,
        batch_frame_idxs,
        file_names,
        batch_center_frame_idxs,
        video_captions_retrieved,
        dataset
    ):
        for idx, frame_idx in enumerate(batch_frame_idxs):
            file_name = file_names[indices[idx][0]] # ex Abuse001_x264.mp4/304
            ret_video_name, ret_index = file_name.split("/")

            if dataset == 'UCF-Crimes':
                ret_video_name = ret_video_name.split('.')[0]

                match = re.match(r'([A-Za-z]+)', ret_video_name)
                if match:
                    class_prefix = match.group(1)  # "Abuse"
                    if class_prefix == 'Normal':
                        class_prefix = 'Testing_Normal_Videos_Anomaly'  # or 'Training_Normal_Videos_Anomaly'
                
                video_caption_path = self.captions_dir_template / class_prefix / ret_video_name / f"{ret_video_name}.json"

            elif dataset == 'XD-Violence':
                video_caption_path = self.captions_dir_template / f"{ret_video_name}.json"

            with open(video_caption_path) as f:
                video_captions = json.load(f)

            center_frame_idx = batch_center_frame_idxs[idx // self.num_samples]
            ret_index = str(f"{int(ret_index):05d}")
            video_captions_retrieved[str(center_frame_idx)][str(frame_idx)] = video_captions[ret_index]

    def _save_results(self, video_name, video_captions_retrieved):
        output_path = self.output_dir / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(video_captions_retrieved, f, indent=4)

def run(
    root_path: str,
    batch_size: int,
    frame_interval: int,
    captions_dir_template: str,
    output_dir: str,
    index_dir: str,
    imagefile_template: str,
    fps: float,
    clip_duration: float,
    num_samples: int,
    num_neighbors: int,
    dataset,
    ref_file
):
    model, preprocess, device = initialize_vlm_model_and_device()

    image_text_caption_cleaner = ImageTextCaptionCleaner(
        model,
        device,
        output_dir,
        num_samples,
        num_neighbors,
        index_dir,
        captions_dir_template,
        clip_duration,
        fps,
        imagefile_template,
        batch_size,
        frame_interval,
        preprocess,
        dataset,
    )
    if dataset=='UCF-Crimes':
        video_list = load_video_ucf(root_path, ref_file) # Train or Test(ucf)
    elif dataset=='XD-Violence':
        video_list = load_video_xd(root_path, ref_file) # or xd_CLIP_rgbtest.csv
    for video in video_list:
        image_text_caption_cleaner.process_video(video, dataset)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pathname", type=str, default="*.json")
    parser.add_argument("--imagefile_template", type=str, default="{0}_frame_{1:05d}.jpg")
    parser.add_argument("--clip_duration", type=float, default=10)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_neighbors", type=int, default=1)
    parser.add_argument("--root_dir", default='../VAD_dataset')
    parser.add_argument("--ref_file")
    parser.add_argument("--dataset")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    root_dir = Path(args.root_dir)
    ref_file = args.ref_file # 이미지 경로 불러올 때 참조할 파일들(ucf : Anomaly_Train(Test), xd : xd_CLIP_rgb(text))

    root_path = root_dir / dataset / "Extracted_Frames"
    captions_dir_template = root_dir / dataset / "Extracted_Frames_captions"
    index_dir = root_dir / dataset / "create_index"
    output_dir = root_dir / dataset / "captions" / "clean"

    fps = 30

    run(
        root_path = root_path,
        batch_size=args.batch_size,
        frame_interval=args.frame_interval,
        output_dir=output_dir,
        captions_dir_template=captions_dir_template,
        index_dir=index_dir,
        imagefile_template=args.imagefile_template,
        fps=fps,
        clip_duration=args.clip_duration,
        num_samples=args.num_samples,
        num_neighbors=args.num_neighbors,
        dataset=dataset,
        ref_file=ref_file
    )