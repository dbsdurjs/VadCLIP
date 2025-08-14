import pandas as pd
import numpy as np

# 1. CSV 경로 설정
csv_path = 'list/ucf_rgb_I3D.csv'  # 실제 경로로 수정하세요

# 2. CSV 읽기
df = pd.read_csv(csv_path)

# 3. 에러 저장용 리스트
error_files = []

# 4. 각 파일 로드하면서 에러 체크
for path in df['path']:
    try:
        _ = np.load(path)  # 데이터는 사용하지 않고, 로드만 시도
    except Exception as e:
        # 모든 종류의 에러(ValueError, FileNotFoundError 등) 잡기
        print(f'[Error] {path}: {e}')
        error_files.append((path, str(e)))

# 5. 에러 요약 출력 및 파일 저장
if error_files:
    print(f'\n총 {len(error_files)}개 파일에서 로드 오류 발생:')
    for p, msg in error_files:
        print(f'  - {p}: {msg}')

    # 원하면 텍스트로도 저장
    with open('load_errors.txt', 'w') as fw:
        for p, msg in error_files:
            fw.write(f'{p}\t{msg}\n')
    print("\n→ load_errors.txt에 에러 목록 저장됨")
else:
    print('\n모든 파일이 정상적으로 로드되었습니다.')
