# import os
# import csv

# def create_new_csv(
#     old_csv='./list/xd_CLIP_rgbtest.csv',
#     new_csv='./list/xd_CLIP_rgbtest_caption.csv',
#     top_dir='/home/yeogeon/YG_main/diffusion_model/VAD_dataset/XD-Violence/xd_caption_feature',
#     subfolders=('1-1004','1005-2004','2005-2804','2805-3319','3320-3954'),
#     train=True
# ):
#     """
#     기존 xd_CLIP-rgb.csv를 참조하여,
#     top_dir 내부의 subfolders들에서 동일 파일명을 가진 npy 파일 경로를 찾아
#     새로운 CSV 파일(new_csv)에 path,label 형식으로 기록한다.
#     """
#     with open(old_csv, 'r', newline='', encoding='utf-8') as f_in, \
#          open(new_csv, 'w', newline='', encoding='utf-8') as f_out:

#         reader = csv.reader(f_in)
#         writer = csv.writer(f_out)

#         # 헤더 작성
#         writer.writerow(["path","label"])

#         # 기존 CSV의 헤더 스킵
#         next(reader, None)

#         for row in reader:
#             if len(row) < 2:
#                 continue  # 혹시 컬럼이 2개 미만이면 스킵
#             old_path, label = row

#             # 예: "/old/path/Bad.Boys...__0.npy" -> "Bad.Boys...__0.npy"
#             filename = os.path.basename(old_path)

#             # 확장자 제거 -> "Bad.Boys...__0"
#             name_no_ext, ext = os.path.splitext(filename)
#             name_no_ext = name_no_ext.rsplit('__', 1)[0]
            
#             # .npy 확장자가 아닐 경우 스킵할지 여부는 상황에 맞게 처리
#             if ext.lower() != '.npy':
#                 continue

#             # subfolders를 순회하며 실제 파일이 있는지 탐색
#             found_path = None

#             if train:
#                 for sf in subfolders:
#                     # 폴더 구조: top_dir/sf/<동영상이름폴더>/<동영상이름>.npy
#                     candidate_dir = os.path.join(top_dir, sf, name_no_ext)
#                     candidate_file = os.path.join(candidate_dir, f'{name_no_ext}.npy')

#                     if os.path.isfile(candidate_file):
#                         found_path = candidate_file
#                         break
#             else:
#                 candidate_dir = os.path.join(top_dir, 'videos', name_no_ext)
#                 candidate_file = os.path.join(candidate_dir, f'{name_no_ext}.npy')

#                 if os.path.isfile(candidate_file):
#                     found_path = candidate_file

#             # 찾았다면 CSV에 기록
#             if found_path is not None:
#                 writer.writerow([found_path, label])
#             else:
#                 # 찾지 못했으면 로그 출력(필요에 따라 처리)
#                 print(f"Warning: {name_no_ext} not found in new structure.")

# if __name__ == '__main__':
#     create_new_csv(train=False)

import os
import shutil

# 원본 경로 (하위에 1-1004, 1005-2004, ... 폴더들이 있음)
src_root = "/home/yeogeon/YG_main/diffusion_model/VAD_dataset/server_dataset/XD-Violence/Extracted_Frames"
# 대상 경로 (모든 txt 파일을 저장할 경로)
dst_dir = "/home/yeogeon/YG_main/diffusion_model/VAD_dataset/server_dataset/XD-Violence/Extracted_Frames_captions"

# 대상 경로가 없으면 생성
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# src_root 하위 폴더를 재귀적으로 탐색
for root, dirs, files in os.walk(src_root):
    for file in files:
        if file.endswith('.txt'):
            src_file = os.path.join(root, file)
            # 원본 파일의 상대 경로를 가져와서, 폴더 구분자를 언더바(_)로 변경하여 접두어로 사용
            rel_path = os.path.relpath(root, src_root)  # 예: "1-1004/동영상폴더명"
            rel_path_modified = rel_path.replace(os.sep, "_")
            dst_file = os.path.join(dst_dir, file)
            shutil.copy2(src_file, dst_file)
            print(f"Copied {src_file} to {dst_file}")
