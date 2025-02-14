import pandas as pd

# 기존 CSV 파일 경로
csv_path = "/home/yeogeon/YG_main/diffusion_model/VadCLIP/list/ucf_CLIP_rgb_caption.csv"  # 예: "list/ucf_CLIP_rgb.csv"

# CSV 파일 읽기
df = pd.read_csv(csv_path)

# 'path' 컬럼이 "__0.npy"로 끝나는 행만 필터링
df_filtered = df[df["path"].str.endswith("__0.npy")]

# 필터링 결과를 새로운 CSV 파일로 저장 (원본을 덮어쓰고 싶다면 csv_path로 저장)
df_filtered.to_csv(csv_path, index=False)

print(f"Filtered CSV 파일이 생성되었습니다: filtered_csv_file.csv")
