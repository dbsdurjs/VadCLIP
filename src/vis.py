import numpy as np
import matplotlib.pyplot as plt
import random

# ✅ 1. GT 파일 로드
gt_file = "/home/yeogeon/YG_main/diffusion_model/VadCLIP/list/Temporal_Anomaly_Annotation.txt"  # GT 파일 경로
anomaly_score_folder = "/path/to/anomaly_scores/"  # 모델 예측값이 저장된 폴더
selected_videos = {}  # 각 클래스에서 1개 동영상만 선택

# 📌 2. GT 데이터 로드 및 각 클래스에서 하나의 동영상 선택
with open(gt_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        video_name, class_name, start_frame, end_frame = parts[0], parts[1], int(parts[2]), int(parts[3])

        # ✅ 각 클래스에서 하나의 동영상만 선택
        if class_name not in selected_videos:
            selected_videos[class_name] = (video_name, start_frame, end_frame)

# 📌 3. 그래프 출력
fig, axes = plt.subplots(2, 4, figsize=(15, 6))  # 2행 4열 그래프
classes = list(selected_videos.keys())

for i, (class_name, (video_name, start_frame, end_frame)) in enumerate(selected_videos.items()):
    ax = axes.flat[i]
    
    # ✅ 모델의 이상 탐지 점수 불러오기
    anomaly_score_path = f"{anomaly_score_folder}/{video_name}.npy"
    try:
        anomaly_scores = np.load(anomaly_score_path)
    except FileNotFoundError:
        anomaly_scores = np.random.rand(2000) * 0.5  # 랜덤 데이터 (테스트용)

    num_frames = len(anomaly_scores)
    
    # 🔴 GT(Ground Truth) 표시
    gt_region = np.zeros(num_frames)
    gt_region[start_frame:end_frame] = 1  # GT 영역을 1로 설정

    ax.fill_between(range(num_frames), 0, 1, where=gt_region > 0, color='red', alpha=0.3)

    # 🔵 모델 예측값 그래프
    ax.plot(anomaly_scores, color="blue", lw=2)

    # 🏷 클래스 및 동영상 이름 추가
    ax.text(num_frames // 2, 0.9, f"{class_name}\n({video_name})", fontsize=12, color="blue", ha='center')

    # ✅ 그래프 설정
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_xlim(0, num_frames)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

# ✅ 4. 결과 저장 및 출력
plt.tight_layout()
plt.savefig("selected_videos_graph.png", dpi=300)
plt.show()
