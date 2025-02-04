import cv2
import numpy as np

import torch
from clip import clip
from PIL import Image
import os

def extract_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
        
    cap.release()
    return np.array(frames)

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

def average_features(features, group_size=16):
    num_features = len(features)
    grouped_features = []
    for i in range(0, num_features, group_size):
        group = features[i:i+group_size]
        if group.shape[0] != group_size:
            continue
        # 그룹의 평균 계산
        grouped_features.append(np.mean(group, axis=0).astype(np.float16))
    return np.array(grouped_features)

if __name__ == '__main__':
    # 1. 동영상 파일들이 저장된 기본 경로 (클래스/동영상.mp4 형식)
    base_path = '/home/yoonyeogeon/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Videos'

    # 2. 추출된 프레임을 이미지 파일로 저장할 경로 (각 클래스별 폴더 생성)
    frames_output_base = '/home/yoonyeogeon/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Extracted_Frames'
    if not os.path.exists(frames_output_base):
        os.makedirs(frames_output_base)

    # 3. CLIP 특징(np.array)을 저장할 경로 (각 클래스별 폴더 생성)
    features_output_base = '/home/yoonyeogeon/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Video_Features'
    if not os.path.exists(features_output_base):
        os.makedirs(features_output_base)

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
        
        # 클래스별 영상 특징 저장 폴더 생성
        features_class_dir = os.path.join(features_output_base, class_name)
        if not os.path.exists(features_class_dir):
            os.makedirs(features_class_dir)
        
        # 6. 클래스 폴더 내의 동영상(mp4) 파일 순회
        for file_name in os.listdir(class_dir):
            if not file_name.lower().endswith('.mp4'):
                continue  # mp4 파일이 아니면 건너뜁니다.
            
            video_path = os.path.join(class_dir, file_name)
            print("Processing video:", video_path)
            
            try:
                # 동영상에서 프레임 추출 (shape: (num_frames, H, W, 3))
                frames = extract_video_frames(video_path)
            except Exception as e:
                print("Error extracting frames from", video_path, e)
                continue
            
            # 동영상 파일명(확장자 제거)
            video_basename = os.path.splitext(file_name)[0]
            
            # 6-1. 추출된 프레임을 이미지 파일로 저장
            # 각 동영상마다 별도의 폴더를 생성하여 프레임들을 저장 (예: video_basename 폴더)
            frames_video_dir = os.path.join(frames_class_dir, video_basename)
            if not os.path.exists(frames_video_dir):
                os.makedirs(frames_video_dir)
            
            for idx, frame in enumerate(frames):
                # 각 프레임을 jpg 파일로 저장
                frame_path = os.path.join(frames_video_dir, f"{video_basename}_frame_{idx:05d}.jpg")
                # PIL을 이용하여 저장 (frame은 numpy array 형식)
                Image.fromarray(frame).save(frame_path)
            print("Saved extracted frames for video", video_path, "to", frames_video_dir)
            
            # 6-2. 동영상 프레임을 CLIP 모델 입력에 맞게 전처리하기 위해 크롭 (예: 224x224)
            corp_video = video_crop(frames, 0)  # corp_video의 shape: (num_frames, 224, 224, 3)
            
            # 6-3. CLIP 모델을 이용해 각 프레임별 특징 추출
            video_features = torch.zeros(0).to(device)
            with torch.no_grad():
                for i in range(corp_video.shape[0]):
                    img = Image.fromarray(corp_video[i])
                    img_tensor = preprocess(img).unsqueeze(0).to(device)
                    feature = model.encode_image(img_tensor)
                    video_features = torch.cat([video_features, feature], dim=0)
            
            # 6-4. 추출한 영상 특징을 numpy 배열로 변환 (shape: (num_frames, 512))
            video_features = video_features.detach().cpu().numpy()
            
            # 6-5. 영상 특징(np.array)을 저장 (동영상 파일명 기반 npy 파일)
            features_output_path = os.path.join(features_class_dir, video_basename + "_features.npy")
            np.save(features_output_path, video_features)
            print("Saved video features for video", video_path, "to", features_output_path)
    
    # -------------------------------------------------------------------------------
    # file1 = './save_path.npy'
    # file2 = '/home/yoonyeogeon/diffusion_model/VAD_dataset/UCFClipFeatures/Abuse/Abuse001_x264__0.npy'
    
    # data1 = np.load(file1)
    # data2 = np.load(file2)

    # final_feature = average_features(data1, group_size=16)
    # print(f"Final feature shape: {final_feature.shape}")  # Shape: [video_frames/16, 512]

    # # 1. 데이터 차원 비교
    # if final_feature.shape != data2.shape:
    #     print(f"Shape mismatch: {final_feature.shape} vs {data2.shape}")
    # else:
    #     print(f"Shapes are identical: {final_feature.shape}")

    # # 2. 데이터 유형 비교
    # if final_feature.dtype != data2.dtype:
    #     print(f"Data type mismatch: {final_feature.dtype} vs {data2.dtype}")
    # else:
    #     print(f"Data types are identical: {final_feature.dtype}")

    # # 3. 값 비교 (element-wise)
    # if np.array_equal(final_feature, data2):
    #     print("The two arrays are identical.")
    # else:
    #     print("The two arrays are not identical.")
    
    # # 4. 차이 확인 (element-wise 차이 계산)
    # diff = final_feature - data2  # 값 차이
    # max_diff = np.max(np.abs(diff))  # 절대값 기준 최대 차이
    # print(f"Maximum difference: {max_diff}")
    
    # # 5. 값이 다른 요소 인덱스 찾기
    # mismatched_indices = np.where(final_feature != data2)
    # print(f"Number of mismatched elements: {len(mismatched_indices[0])}")
    # print(f"Example mismatched indices: {mismatched_indices[0][:10]}")  # 최대 10개만 표시

    # # 6. 데이터의 요약 통계량 비교
    # print("\nSummary statistics for file1:")
    # print(f"Mean: {np.mean(final_feature)}, Min: {np.min(final_feature)}, Max: {np.max(final_feature)}")

    # print("\nSummary statistics for file2:")
    # print(f"Mean: {np.mean(data2)}, Min: {np.min(data2)}, Max: {np.max(data2)}")