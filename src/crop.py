import cv2
import numpy as np

import torch
from clip import clip
from PIL import Image

def extract_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
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
    # video = np.zeros([3, 320, 240, 3], dtype=np.uint8)
    # video_path = '/home/yoonyeogeon/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Videos/Abuse/Abuse001_x264.mp4'
    # frames = extract_video_frames(video_path)  # (프레임 개수, h, w, 3)
    # corp_video = video_crop(frames, 0)  # (프레임 개수, 224, 224, 3)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/16", device)
    # video_features = torch.zeros(0).to(device)
    # with torch.no_grad():
    #     for i in range(frames.shape[0]):
    #         img = Image.fromarray(corp_video[i])
    #         img = preprocess(img).unsqueeze(0).to(device)
    #         feature = model.encode_image(img)
    #         video_features = torch.cat([video_features, feature], dim=0)
    
    # video_features = video_features.detach().cpu().numpy()  # (총 프레임 개수, 512)
    # np.save('save_path', video_features)
    
    # -------------------------------------------------------------------------------
    file1 = './save_path.npy'
    file2 = '/home/yoonyeogeon/diffusion_model/VAD_dataset/UCFClipFeatures/Abuse/Abuse001_x264__0.npy'
    
    data1 = np.load(file1)
    data2 = np.load(file2)

    final_feature = average_features(data1, group_size=16)
    print(f"Final feature shape: {final_feature.shape}")  # Shape: [video_frames/16, 512]

    # 1. 데이터 차원 비교
    if final_feature.shape != data2.shape:
        print(f"Shape mismatch: {final_feature.shape} vs {data2.shape}")
    else:
        print(f"Shapes are identical: {final_feature.shape}")

    # 2. 데이터 유형 비교
    if final_feature.dtype != data2.dtype:
        print(f"Data type mismatch: {final_feature.dtype} vs {data2.dtype}")
    else:
        print(f"Data types are identical: {final_feature.dtype}")

    # 3. 값 비교 (element-wise)
    if np.array_equal(final_feature, data2):
        print("The two arrays are identical.")
    else:
        print("The two arrays are not identical.")
    
    # 4. 차이 확인 (element-wise 차이 계산)
    diff = final_feature - data2  # 값 차이
    max_diff = np.max(np.abs(diff))  # 절대값 기준 최대 차이
    print(f"Maximum difference: {max_diff}")
    
    # 5. 값이 다른 요소 인덱스 찾기
    mismatched_indices = np.where(final_feature != data2)
    print(f"Number of mismatched elements: {len(mismatched_indices[0])}")
    print(f"Example mismatched indices: {mismatched_indices[0][:10]}")  # 최대 10개만 표시

    # 6. 데이터의 요약 통계량 비교
    print("\nSummary statistics for file1:")
    print(f"Mean: {np.mean(final_feature)}, Min: {np.min(final_feature)}, Max: {np.max(final_feature)}")

    print("\nSummary statistics for file2:")
    print(f"Mean: {np.mean(data2)}, Min: {np.min(data2)}, Max: {np.max(data2)}")