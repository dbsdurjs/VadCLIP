from pathlib import Path
import shutil
import argparse
import numpy as np
import time
import ffmpeg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from extract_features import run
from resnet import i3_res50
import os

# resnet i3d model로 npy 파일 추출

# def generate(datasetpath, outputpath, pretrainedpath, frequency, batch_size, sample_mode):
# 	Path(outputpath).mkdir(parents=True, exist_ok=True)
# 	temppath = outputpath+ "/temp/"
# 	rootdir = Path(datasetpath)
# 	videos = [str(f) for f in rootdir.glob('**/*.mp4')]
# 	# setup the model
# 	i3d = i3_res50(400, pretrainedpath)
# 	i3d.cuda()
# 	i3d.train(False)  # Set model to evaluate mode
# 	for video in videos:
# 		videoname = video.split("/")[-1].rsplit(".", 1)[0]
# 		startime = time.time()
# 		print("Generating for {0}".format(video))
# 		Path(temppath).mkdir(parents=True, exist_ok=True)
# 		ffmpeg.input(video).output('{}%d.jpg'.format(temppath),start_number=0).global_args('-loglevel', 'quiet').run()
# 		print("Preprocessing done..")
# 		features = run(i3d, frequency, temppath, batch_size, sample_mode)
# 		print("Obtained features of size: ", features.shape)

# 		# sample_mode에 따라 저장 방식을 분기
# 		if sample_mode == 'oversample':
# 			# features.shape: (num_chunks, 10, feature_dim)
# 			num_crops = features.shape[1]
# 			for crop_idx in range(num_crops):
# 				crop_features = features[:, crop_idx, :]  # 각 crop별 feature, shape: (num_chunks, feature_dim)
# 				output_file = os.path.join(outputpath, f"{videoname}__{crop_idx}.npy")
# 				np.save(output_file, crop_features)
# 		else:  # center_crop 모드
# 			# features.shape: (num_chunks, 1, feature_dim)
# 			output_file = os.path.join(outputpath, f"{videoname}.npy")
# 			np.save(output_file, features[:, 0, :])

# 		shutil.rmtree(temppath)
# 		print("done in {0}.".format(time.time() - startime))
def generate(datasetpath, outputpath, pretrainedpath,
             frequency, batch_size, sample_mode,
             error_list_file=None):
    """
    datasetpath: 비디오(.mp4)들이 있는 루트 디렉터리
    outputpath: npy가 저장될 디렉터리
    error_list_file: 재추출할 npy 경로들이 담긴 텍스트 파일 (한 줄에 하나)
    """
    Path(outputpath).mkdir(parents=True, exist_ok=True)
    temppath = os.path.join(outputpath, "temp")
    rootdir = Path(datasetpath)

    # 1) error_list_file이 있으면, videoname → [crop_idx,...] 매핑 생성
    error_videos = {}  # { videoname(str) : [crop_idx(int), ...] }
    if error_list_file:
        with open(error_list_file, 'r') as f:
            for line in f:
                path = line.strip().split()[0]  # "npy_path  오류메시지" 면 첫 토큰
                base = os.path.basename(path)
                # "__{crop}.npy" 패턴이면 split
                if "__" in base:
                    videoname, crop = base.rsplit("__", 1)
                    crop_idx = int(os.path.splitext(crop)[0])
                else:
                    # 만약 single-crop 모드라 crop 정보가 없으면 None 처리
                    videoname = os.path.splitext(base)[0]
                    crop_idx = None
                error_videos.setdefault(videoname, []).append(crop_idx)

    # 2) 처리할 비디오 리스트 구성
    if error_videos:
        videos = []
        for videoname in error_videos:
            # datasetpath 하위에서 videoname.mp4 찾기
            matches = list(rootdir.glob(f'**/{videoname}.mp4'))
            if not matches:
                print(f"[Warning] 비디오 파일을 찾을 수 없음: {videoname}.mp4")
            else:
                videos.append(str(matches[0]))
    else:
        # 에러 리스트 없으면 전체 비디오 처리
        videos = [str(f) for f in rootdir.glob('**/*.mp4')]

    # 3) 모델 로드
    i3d = i3_res50(400, pretrainedpath)
    i3d.cuda()
    i3d.train(False)

    # 4) 비디오별 feature 추출
    for video in videos:
        videoname = Path(video).stem
        startime = time.time()
        print(f"\n>>> Generating for {videoname}.mp4")
        # temp 경로 초기화
        Path(temppath).mkdir(parents=True, exist_ok=True)
        ffmpeg.input(video)\
              .output(f'{temppath}/%d.jpg', start_number=0)\
              .global_args('-loglevel', 'quiet')\
              .run()

        # I3D로 feature 추출
        features = run(i3d, frequency, temppath, batch_size, sample_mode)
        print("  Obtained features:", features.shape)

        # 저장
        if sample_mode == 'oversample':
            # 에러 리스트가 있으면 거기에 명시된 crop만, 아니면 전체 crop
            crops = error_videos.get(videoname, range(features.shape[1]))
            for crop_idx in crops:
                if crop_idx is None or crop_idx >= features.shape[1]:
                    print(f"  [Skip] invalid crop_idx={crop_idx} for {videoname}")
                    continue
                out_feat = features[:, crop_idx, :]  # (num_chunks, feature_dim)
                out_file = os.path.join(outputpath, f"{videoname}__{crop_idx}.npy")
                np.save(out_file, out_feat)
                print(f"  Saved: {os.path.basename(out_file)}")

        else:  # center_crop 모드
            # 에러 리스트에 있으면 저장, 아니면 무시
            if videoname in error_videos:
                out_feat = features[:, 0, :]
                out_file = os.path.join(outputpath, f"{videoname}.npy")
                np.save(out_file, out_feat)
                print(f"  Saved center crop: {os.path.basename(out_file)}")
            else:
                print(f"  [Skip] {videoname} (no errors listed)")

        # temp 정리
        shutil.rmtree(temppath)
        print(f"  Done in {time.time() - startime:.1f}s")
        
if __name__ == '__main__': 
	parser = argparse.ArgumentParser()
	parser.add_argument('--datasetpath', type=str, default="../VAD_dataset/UCF-Crimes/train_videos/")
	parser.add_argument('--outputpath', type=str, default="../VAD_dataset/UCFClipFeatures_I3D/")
	parser.add_argument('--pretrainedpath', type=str, default="./I3D_Feature_Extraction_resnet/pretrained/i3d_r50_kinetics.pth")
	parser.add_argument('--frequency', type=int, default=16)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--sample_mode', type=str, default="oversample")
	parser.add_argument('--error_list',    type=str, default="load_errors.txt",
                        help="재추출할 npy 경로들이 담긴 텍스트 파일")
	args = parser.parse_args()
	generate(args.datasetpath, str(args.outputpath), args.pretrainedpath, args.frequency, args.batch_size, args.sample_mode, error_list_file=args.error_list)    
