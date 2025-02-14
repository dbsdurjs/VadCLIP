import os
import pandas as pd

# 기존 CSV 파일 경로
csv_path = "/home/yeogeon/YG_main/diffusion_model/VadCLIP/list/ucf_CLIP_rgb_caption.csv"  # 예: "list/ucf_CLIP_rgb.csv"

# CSV 파일 읽기
df = pd.read_csv(csv_path)

# 각 행의 'path' 열을 수정
def modify_path(old_path):
    # old_path 예: 
    # /.../all_ucfclip_caption_feature/Abuse/Abuse001_x264.npy
    dir_path, filename = os.path.split(old_path)  # dir_path: .../Abuse, filename: Abuse001_x264.npy
    basename, ext = os.path.splitext(filename)    # basename: Abuse001_x264, ext: .npy
    # 새 경로: 기존 폴더 안에 basename 폴더를 만들고, 그 안에 파일을 위치
    new_path = os.path.join(dir_path, basename, filename)
    return new_path

# 'path' 열에 적용
df["path"] = df["path"].apply(modify_path)

# 새 CSV 파일로 저장
df.to_csv(csv_path, index=False)
print(f"새 CSV 파일이 생성되었습니다: {csv_path}")
