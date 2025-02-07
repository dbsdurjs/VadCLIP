# 동영상에서 프레임 추출 및 저장
import cv2
import numpy as np

import torch
from clip import clip
from PIL import Image
import os

def extract_video_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # ✅ BGR → RGB 변환
        frame_pil = Image.fromarray(frame_rgb)

        frame_path = os.path.join(output_dir, f"{video_basename}_frame_{frame_count:05d}.jpg")
        frame_pil.save(frame_path)

        frame_count += 1
        
    cap.release()
    print(f"✅ {video_path}: {frame_count}개의 프레임 저장 완료")

    return True

def video_crop(video_frame, type):
    l = video_frame.shape[0]
    new_frame = []
    for i in range(l):
        img = cv2.resize(video_frame[i], dsize=(340, 256))
        new_frame.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    #1
    img = np.array(new_frame)
    if type == 0:
        img = img[:, 16:240, 58:282, :]
    #2
    elif type == 1:
        img = img[:, :224, :224, :]
    #3
    elif type == 2:
        img = img[:, :224, -224:, :]
    #4
    elif type == 3:
        img = img[:, -224:, :224, :]
    #5
    elif type == 4:
        img = img[:, -224:, -224:, :]
    #6
    elif type == 5:
        img = img[:, 16:240, 58:282, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #7
    elif type == 6:
        img = img[:, :224, :224, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #8
    elif type == 7:
        img = img[:, :224, -224:, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #9
    elif type == 8:
        img = img[:, -224:, :224, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #10
    elif type == 9:
        img = img[:, -224:, -224:, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    
    return img

def image_crop(image, type):
    img = cv2.resize(image, dsize=(340, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #1
    if type == 0:
        img = img[16:240, 58:282, :]
    #2
    elif type == 1:
        img = img[:224, :224, :]
    #3
    elif type == 2:
        img = img[:224, -224:, :]
    #4
    elif type == 3:
        img = img[-224:, :224, :]
    #5
    elif type == 4:
        img = img[-224:, -224:, :]
    #6
    elif type == 5:
        img = img[16:240, 58:282, :]
        img = cv2.flip(img, 1)
    #7
    elif type == 6:
        img = img[:224, :224, :]
        img = cv2.flip(img, 1)
    #8
    elif type == 7:
        img = img[:224, -224:, :]
        img = cv2.flip(img, 1)
    #9
    elif type == 8:
        img = img[-224:, :224, :]
        img = cv2.flip(img, 1)
    #10
    elif type == 9:
        img = img[-224:, -224:, :]
        img = cv2.flip(img, 1)
    
    return img

if __name__ == '__main__':
    # 1. 동영상 파일들이 저장된 기본 경로 (클래스/동영상.mp4 형식)
    base_path = '/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Videos'

    # 2. 추출된 프레임을 이미지 파일로 저장할 경로 (각 클래스별 폴더 생성)
    frames_output_base = '/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Extracted_Frames'
    if not os.path.exists(frames_output_base):
        os.makedirs(frames_output_base)

    # 4. CLIP 모델 및 전처리기 로드 (ViT-B/16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device)

    # 5. base_path 하위의 각 클래스 폴더 순회
    for class_name in os.listdir(base_path):
        class_dir = os.path.join(base_path, class_name)
        if not os.path.isdir(class_dir):
            continue  # 디렉토리가 아니면 건너뜁니다.
        
        # 클래스별 추출 프레임 저장 폴더 생성
        frames_class_dir = os.path.join(frames_output_base, class_name)
        if not os.path.exists(frames_class_dir):
            os.makedirs(frames_class_dir)
        
        exclude_file = ['Explosion046_x264.mp4', 'Arson019_x264.mp4', 'Normal_Videos_935_x264.mp4', 'Normal_Videos308_x264.mp4', 'Normal_Videos307_x264.mp4',
                        'Normal_Videos633_x264.mp4', 'Normal_Videos946_x264.mp4', 'Normal_Videos471_x264.mp4', 'Normal_Videos947_x264.mp4', 'Normal_Videos472_x264.mp4',
                        'Normal_Videos425_x264.mp4', 'Normal_Videos547_x264.mp4', 'Normal_Videos138_x264.mp4', 'Normal_Videos529_x264.mp4', 'Normal_Videos530_x264.mp4',
                        'Normal_Videos450_x264.mp4', 'Normal_Videos666_x264.mp4', 'Normal_Videos449_x264.mp4']  # 용량 큰 파일들
        
        # 6. 클래스 폴더 내의 동영상(mp4) 파일 순회
        for file_name in os.listdir(class_dir):
            if not file_name.lower().endswith('.mp4'):
                continue  # mp4 파일이 아니면 건너뜁니다.
            if file_name not in exclude_file:   # 용량 큰 파일 제외
                continue
            video_path = os.path.join(class_dir, file_name)
            print("Processing video:", video_path)
            
            # 동영상 파일명(확장자 제거)
            video_basename = os.path.splitext(file_name)[0]
            
            # 6-1. 추출된 프레임을 이미지 파일로 저장
            # 각 동영상마다 별도의 폴더를 생성하여 프레임들을 저장 (예: video_basename 폴더)
            frames_video_dir = os.path.join(frames_class_dir, video_basename)
            if not os.path.exists(frames_video_dir):
                os.makedirs(frames_video_dir)

            try:
                # 동영상에서 프레임 추출 (shape: (num_frames, H, W, 3))
                frames = extract_video_frames(video_path, frames_video_dir)
            except Exception as e:
                print("Error extracting frames from", video_path, e)
                continue