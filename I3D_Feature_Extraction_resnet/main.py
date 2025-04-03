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

def generate(datasetpath, outputpath, pretrainedpath, frequency, batch_size, sample_mode):
	Path(outputpath).mkdir(parents=True, exist_ok=True)
	temppath = outputpath+ "/temp/"
	rootdir = Path(datasetpath)
	videos = [str(f) for f in rootdir.glob('**/*.mp4')]
	# setup the model
	i3d = i3_res50(400, pretrainedpath)
	i3d.cuda()
	i3d.train(False)  # Set model to evaluate mode
	for video in videos:
		videoname = video.split("/")[-1].rsplit(".", 1)[0]
		startime = time.time()
		print("Generating for {0}".format(video))
		Path(temppath).mkdir(parents=True, exist_ok=True)
		ffmpeg.input(video).output('{}%d.jpg'.format(temppath),start_number=0).global_args('-loglevel', 'quiet').run()
		print("Preprocessing done..")
		features = run(i3d, frequency, temppath, batch_size, sample_mode)
		print("Obtained features of size: ", features.shape)

		# sample_mode에 따라 저장 방식을 분기
		if sample_mode == 'oversample':
			# features.shape: (num_chunks, 10, feature_dim)
			num_crops = features.shape[1]
			for crop_idx in range(num_crops):
				crop_features = features[:, crop_idx, :]  # 각 crop별 feature, shape: (num_chunks, feature_dim)
				output_file = os.path.join(outputpath, f"{videoname}__{crop_idx}.npy")
				np.save(output_file, crop_features)
		else:  # center_crop 모드
			# features.shape: (num_chunks, 1, feature_dim)
			output_file = os.path.join(outputpath, f"{videoname}.npy")
			np.save(output_file, features[:, 0, :])

		shutil.rmtree(temppath)
		print("done in {0}.".format(time.time() - startime))

if __name__ == '__main__': 
	parser = argparse.ArgumentParser()
	parser.add_argument('--datasetpath', type=str, default="../VAD_dataset/XD-Violence/train_videos/")
	parser.add_argument('--outputpath', type=str, default="../VAD_dataset/XDClipFeatures_I3D/")
	parser.add_argument('--pretrainedpath', type=str, default="./I3D_Feature_Extraction_resnet/pretrained/i3d_r50_kinetics.pth")
	parser.add_argument('--frequency', type=int, default=16)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--sample_mode', type=str, default="oversample")
	args = parser.parse_args()
	generate(args.datasetpath, str(args.outputpath), args.pretrainedpath, args.frequency, args.batch_size, args.sample_mode)    
