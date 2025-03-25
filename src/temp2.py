import os
import numpy as np
import pandas as pd
import csv
def match_filename(path: str, xd_cap = None) -> str:
    """
    주어진 파일 경로에서 확장자를 제외한 순수 파일 이름을 반환.
    예) /path/to/Normal_Videos948_x264__0.npy -> Normal_Videos948_x264__0
    """
    if xd_cap is None:
        return os.path.splitext(os.path.basename(path))[0].rsplit('__',1)[0]
    else:
        return os.path.splitext(os.path.basename(path))[0]

def align_features(clip_feat: np.ndarray, cap_feat: np.ndarray):
    """
    clip_feat과 cap_feat의 첫 번째 차원(프레임 수)이 다르면 맞춰주는 함수.
    - cap_feat이 더 크면 앞쪽부터 clip_feat.shape[0]만 남김.
    - cap_feat이 더 작으면 마지막 프레임을 반복해서 clip_feat.shape[0]만큼 확장.
    - 두 배열의 임베딩 차원(D)는 같다고 가정.
    """
    n_clip = clip_feat.shape[0]
    n_cap = cap_feat.shape[0]

    if n_clip == n_cap:
        # 이미 길이가 동일하면 그대로
        return clip_feat, cap_feat
    elif n_cap > n_clip:
        # 캡션 배열이 더 길면 앞쪽에서 n_clip만큼만 남긴다
        diff = n_cap - n_clip
        cap_feat = cap_feat[:-diff]
        return clip_feat, cap_feat
    else:
        # 캡션 배열이 더 짧으면 마지막 프레임을 복제해 길이를 맞춘다
        diff = n_clip - n_cap
        last_frame = cap_feat[-1:]
        pad_array = np.repeat(last_frame, diff, axis=0)
        cap_feat = np.concatenate([cap_feat, pad_array], axis=0)
        return clip_feat, cap_feat

def main(first_csv: str, second_csv: str):
    # 첫 번째 CSV: 클립 특징 정보
    df1 = pd.read_csv(first_csv)
    # 두 번째 CSV: 캡션 특징 정보
    df2 = pd.read_csv(second_csv)

    # 파일명(확장자 제거)을 키로 하여 매핑
    df1["key"] = df1["path"].apply(match_filename)
    df2["key"] = df2["path"].apply(match_filename, xd_cap = True)

    # inner join으로 매칭되는 파일만 연결
    # (매칭 안 되는 파일이 있으면 제외됨)
    df_merged = pd.merge(df1, df2, on="key", how="inner", suffixes=("_clip", "_cap"))

    print(f"총 {len(df_merged)}개가 매칭되었습니다.")

    # 매칭된 각 쌍에 대해 shape을 맞춘 뒤, 원하는 처리를 수행
    for i, row in df_merged.iterrows():
        clip_path = row["path_clip"]
        cap_path = row["path_cap"]
        label_clip = row["label_clip"]
        label_cap = row["label_cap"]

        # npy 파일 로드
        clip_feat = np.load(clip_path)
        cap_feat = np.load(cap_path)

        # 두 배열의 shape이 (프레임 수, 임베딩 차원)인지 확인 (예: (N, 512))
        if clip_feat.ndim != 2 or cap_feat.ndim != 2:
            print(f"[경고] 2차원 배열이 아닙니다. clip:{clip_feat.shape}, cap:{cap_feat.shape}")
            continue

        # 프레임 길이 맞추기
        clip_feat_aligned, cap_feat_aligned = align_features(clip_feat, cap_feat)

        # 여기서부터는 aligned된 데이터를 활용
        # 예: 그대로 넘파이로 저장, 또는 다른 로직에 전달
        # 아래는 예시로 임시 파일에 저장하는 코드
        # (원하는 경로로 수정)
        base_key = row["key"]
        np.save(f"/home/yeogeon/YG_main/diffusion_model/VAD_dataset/XD-Violence/aligned_output_xd/{base_key}.npy", clip_feat_aligned)
        np.save(f"/home/yeogeon/YG_main/diffusion_model/VAD_dataset/XD-Violence/aligned_output_xd/{base_key}.npy", cap_feat_aligned)

        # 필요하다면 label도 csv에 기록하거나, 별도 리스트에 모아두기
        # ...


def rewrite_csv(old_csv, new_csv):
    """
    old_csv 파일을 읽어서, 
    '/home/.../aligned_output_xd/서브폴더/파일.npy,label'
    형태를
    '/home/.../aligned_output_xd/파일.npy,label'
    로 바꾼 뒤, new_csv에 저장.
    """
    with open(old_csv, 'r', newline='', encoding='utf-8') as fin, \
         open(new_csv, 'w', newline='', encoding='utf-8') as fout:
        
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        
        for row in reader:
            # row는 [path, label] 구조라고 가정
            if len(row) < 2:
                continue
            
            old_path, label = row[0], row[1]

            # 1) 경로에서 파일명 추출
            filename = os.path.basename(old_path)  
            # 예: "Bad.Boys.1995__#00-26-51_00-27-53_label_B2-0-0.npy"

            # 2) 상위 디렉터리 추출: 
            #    old_path = ".../aligned_output_xd/폴더/파일.npy"
            #    os.path.dirname(old_path) = ".../aligned_output_xd/폴더"
            #    os.path.dirname(os.path.dirname(old_path)) = ".../aligned_output_xd"
            upper_dir = os.path.dirname(os.path.dirname(old_path))
            # 예: ".../aligned_output_xd"

            # 3) 새 경로 구성
            new_path = os.path.join(upper_dir, filename)
            # 예: ".../aligned_output_xd/Bad.Boys.1995__#00-26-51_00-27-53_label_B2-0-0.npy"

            # 4) CSV에 기록
            writer.writerow([new_path, label])

if __name__ == "__main__":
    first_csv_path = "./list/xd_CLIP_rgb.csv"
    second_csv_path = "./list/xd_CLIP_rgbtest_caption.csv"
    s_csv_path = "./list/xd_CLIP_rgbtest_caption_1.csv"

    rewrite_csv(second_csv_path, s_csv_path)
    # main(first_csv_path, second_csv_path)
