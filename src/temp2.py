from pathlib import Path
import csv
import re

# 1) 탐색할 최상위 폴더
root = Path("../VAD_dataset/UCFClipFeatures_I3D/test")

# 2) 결과를 기록할 CSV 파일 경로
output_csv = Path("paths_labels.csv")

# 3) 파일명에서 순수 알파벳 레이블만 추출하는 정규표현식
#    - 맨 앞부터 알파벳(A–Z, a–z) 연속 구간만 매칭
label_pattern = re.compile(r'^[A-Za-z]+')

with output_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label"])

    for npy_path in root.rglob("*.npy"):
        stem = npy_path.stem  # e.g. "Normal_Videos504_x264__0"
        
        m = label_pattern.match(stem)
        if m:
            label = m.group(0)  # "Normal", "Burglary", "RoadAccidents", ...
        else:
            label = "Unknown"

        writer.writerow([str(npy_path), label])

print(f"✅ 저장 완료: {output_csv.resolve()}")

# from pathlib import Path
# import csv
# import re

# # 1) 탐색할 최상위 폴더
# root = Path("../VAD_dataset/XDClipFeatures_I3D/test")

# # 2) 결과를 기록할 CSV 파일 경로
# output_csv = Path("xd_paths_labels.csv")
# # 3) 파일명에서 '_label_' 뒤, '__' 앞부분을 레이블로 추출하는 정규표현식
# label_pattern = re.compile(r"_label_(.+?)__")
# # 4) (경로, 레이블) 리스트에 수집
# entries = []
# for npy_path in root.rglob("*.npy"):
#     fname = npy_path.name
#     m = label_pattern.search(fname)
#     label = m.group(1) if m else "Unknown"
#     entries.append((npy_path, label))

# # 5) 레이블 오름차순 → 동일 레이블 내에서는 파일명 오름차순으로 정렬
# entries.sort(key=lambda x: (x[1], x[0].name))

# # 6) 정렬된 리스트를 CSV로 기록
# with output_csv.open("w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["path", "label"])
#     for path, label in entries:
#         writer.writerow([str(path), label])

# print(f"✅ 저장 완료: {output_csv.resolve()}")
