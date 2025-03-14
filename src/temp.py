import os

# 작업할 폴더 경로 지정
folder_path = '/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/create_index'

# 폴더 내 모든 파일에 대해 반복 처리
for filename in os.listdir(folder_path):
    # 파일명에 '.mp4'가 포함되어 있다면
    if '.mp4' in filename:
        # '.mp4'를 제거한 새로운 파일명 생성
        new_filename = filename.replace('.mp4', '')
        # 원본 파일 경로와 변경될 파일 경로 지정
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        
        # 파일명 변경 실행
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")
