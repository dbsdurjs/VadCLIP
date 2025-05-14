import os
from pathlib import Path

def count_images_with_patterns(root_dir: str,
                               exts: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.gif'),
                               pattern_labelA: str = 'label_A',
                               pattern_normal: str = 'Normal') -> tuple:
    """
    root_dir 폴더 이하에서 다음을 카운트하여 반환합니다.
      1) 이름이 pattern_labelA로 끝나는 폴더 내 이미지 수
      2) 이름에 pattern_normal이 포함되는 폴더 내 이미지 수
      3) 전체 이미지 수
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise ValueError(f"'{root_dir}' 경로가 존재하지 않거나 디렉토리가 아닙니다.")

    total_labelA = 0
    total_normal = 0
    total_all = 0

    # 모든 파일을 재귀 탐색
    for img_path in root_path.rglob('*'):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in exts:
            continue

        total_all += 1

        # 상위 폴더 중 하나라도 pattern_labelA로 끝나면 카운트
        if any(parent.name.endswith(pattern_labelA) for parent in img_path.parents):
            total_labelA += 1

        # 상위 폴더 중 하나라도 pattern_normal을 포함하면 카운트
        if any(pattern_normal in parent.name for parent in img_path.parents):
            total_normal += 1

    return total_labelA, total_normal, total_all

if __name__ == '__main__':
    # 실제 경로로 변경하세요.
    root_directory = '../VAD_dataset/UCF-Crimes/Extracted_Frames'

    count_labelA, count_normal, count_all = count_images_with_patterns(root_directory)
    print(f"▶ 'label_A'로 끝나는 폴더 내 이미지 개수: {count_labelA}")
    print(f"▶ 'Normal'을 포함하는 폴더 내 이미지 개수: {count_normal}")
    print(f"▶ 전체 이미지 개수:               {count_all}")
