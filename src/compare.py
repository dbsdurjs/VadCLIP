# 추출한 프레임과 기존 제공한 feature npy 파일의 크기 비교 코드
import numpy as np

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
    # -------------------------------------------------------------------------------
    # file1 = './save_path.npy'
    file2 = '/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCFClipFeatures/Fighting/Fighting020_x264__0.npy'
    data2 = np.load(file2)
    print(data2.shape)
    
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