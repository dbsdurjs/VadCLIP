import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('./list/ucf_CLIP_rgbtest_description.csv')  # 파일명을 실제로 사용하시는 이름으로 변경하세요

# 정규표현식으로 경로 수정
df['path'] = df['path'].str.replace(
    r'(x264)(\.npy)$',
    r'\1_captions_janus_pro\2',
    regex=True
)

# 변경된 내용을 새 CSV로 저장
df.to_csv('output.csv', index=False)
