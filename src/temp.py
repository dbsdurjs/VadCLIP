import os
import shutil

def copy_video_txt_files(source_root, target_root):
    """
    source_root: 동영상 파일들이 포함된 최상위 폴더
    target_root: 복사한 파일들을 저장할 대상 폴더 (존재하지 않으면 생성)
    """
    # 대상 폴더 생성 (없으면 생성)
    os.makedirs(target_root, exist_ok=True)

    # 소스 폴더 내를 재귀적으로 순회
    for dirpath, dirnames, filenames in os.walk(source_root):
        for filename in filenames:
            # .txt 파일인 경우만 고려
            if filename.endswith(".txt"):
                # 현재 파일의 경로
                file_path = os.path.join(dirpath, filename)
                # 현재 파일이 위치한 폴더 이름
                folder_name = os.path.basename(dirpath)
                # 파일 이름이 폴더 이름과 동일한지 확인 (예: "Abuse001_x264.txt")
                if filename == f"{folder_name}.txt":
                    # 대상 경로: 원본의 상대 경로를 유지하면서 복사할 수 있음
                    rel_path = os.path.relpath(dirpath, source_root)
                    dest_dir = os.path.join(target_root, rel_path)
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_file = os.path.join(dest_dir, filename)
                    shutil.copy2(file_path, dest_file)
                    print(f"Copied {file_path} to {dest_file}")

if __name__ == '__main__':
    # 예시 경로 (필요에 맞게 수정)
    source_root = "/media/vcl/DATA/YG/Extracted_Frames/"
    target_root = "/media/vcl/DATA/YG/Extracted_Frames/Extracted_Frames_captions/"

    copy_video_txt_files(source_root, target_root)
