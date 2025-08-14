from pathlib import Path
import csv
import shutil

# ——————————————————————————————
# 설정 부분만 필요에 따라 수정하세요
# CSV 파일 위치
csv_file = Path("list/xd_rgb_I3D.csv")

# 원본 파일들이 들어있는 디렉토리 (이 안에서 파일명만 매칭)
src_dir = Path("../VAD_dataset/XDClipFeatures_I3D/train")

# 옮겨갈 대상 디렉토리 (train의 상위 디렉토리)
dst_dir = Path("../VAD_dataset/XDClipFeatures_I3D")
# ——————————————————————————————

# CSV 읽기
with csv_file.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # CSV의 path 필드에서 파일명만 꺼내기
        filename = Path(row["path"]).name
        
        # train 폴더 내 실제 경로 생성
        src_path = src_dir / filename
        
        if src_path.exists():
            # 이동
            dest_path = dst_dir / filename
            shutil.move(str(src_path), str(dest_path))
            print(f"[OK]  {src_path} → {dest_path}")
        else:
            print(f"[SKIP] 파일 없음: {src_path}")
