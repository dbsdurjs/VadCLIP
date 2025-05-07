import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('./list/xd_CLIP_rgbtest_description.csv')  # 실제 경로로 변경

# .npy 확장자 앞에 _captions_janus_pro 삽입
df['path'] = df['path'].str.replace(
    r'\.npy$',               # 문자열 끝의 ".npy"를
    '_captions_janus_pro.npy',  # "_captions_janus_pro.npy"로 대체
    regex=True
)

# 변경된 내용을 새 CSV로 저장
df.to_csv('output.csv', index=False)
