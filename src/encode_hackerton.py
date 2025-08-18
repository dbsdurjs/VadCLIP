import os
import re
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from clip import clip  # pip install git+https://github.com/openai/CLIP.git

# -----------------------------
# 원본 크롭/플립 로직 유지
# -----------------------------
def image_crop(image, type):
    img = cv2.resize(image, dsize=(340, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #1
    if type == 0:
        img = img[16:240, 58:282, :]
    #2
    elif type == 1:
        img = img[:224, :224, :]
    #3
    elif type == 2:
        img = img[:224, -224:, :]
    #4
    elif type == 3:
        img = img[-224:, :224, :]
    #5
    elif type == 4:
        img = img[-224:, -224:, :]
    #6
    elif type == 5:
        img = img[16:240, 58:282, :]
        img = cv2.flip(img, 1)
    #7
    elif type == 6:
        img = img[:224, :224, :]
        img = cv2.flip(img, 1)
    #8
    elif type == 7:
        img = img[:224, -224:, :]
        img = cv2.flip(img, 1)
    #9
    elif type == 8:
        img = img[-224:, :224, :]
        img = cv2.flip(img, 1)
    #10
    elif type == 9:
        img = img[-224:, -224:, :]
        img = cv2.flip(img, 1)
    return img

# -----------------------------
# 유틸: 자연 정렬 (frame_1, frame_2, ..., frame_10)
# -----------------------------
_num_pat = re.compile(r'(\d+)')
def natural_key(path):
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in _num_pat.split(name)]

# -----------------------------
# (T,512) -> (T/snippet_len,512) 변환
# remainder='pad' : 마지막 벡터 반복 패딩
# remainder='drop': 나머지 버림
# pool='mean' 또는 'max'
# -----------------------------
def pool_snippets(feats: np.ndarray, snippet_len: int = 16,
                  pool: str = 'mean', remainder: str = 'pad') -> np.ndarray:
    assert feats.ndim == 2, "feats는 (T, D) 여야 합니다."
    T, D = feats.shape
    if snippet_len <= 0:
        raise ValueError("snippet_len은 1 이상이어야 합니다.")

    if remainder == 'pad':
        pad = (-T) % snippet_len  # multiple 맞추기 위해 필요한 패딩 길이
        if pad > 0:
            last = feats[-1:, :].repeat(pad, axis=0)
            feats_proc = np.concatenate([feats, last], axis=0)
        else:
            feats_proc = feats
    elif remainder == 'drop':
        keep = T - (T % snippet_len)
        feats_proc = feats[:keep, :]
    else:
        raise ValueError("remainder는 'pad' 또는 'drop' 중 하나여야 합니다.")

    if feats_proc.shape[0] == 0:
        return np.zeros((0, D), dtype=feats.dtype)

    # (N*L, D) -> (N, L, D)
    N = feats_proc.shape[0] // snippet_len
    feats_proc = feats_proc.reshape(N, snippet_len, D)

    if pool == 'mean':
        out = feats_proc.mean(axis=1)
    elif pool == 'max':
        out = feats_proc.max(axis=1)
    else:
        raise ValueError("pool은 'mean' 또는 'max' 중 하나여야 합니다.")

    return out  # (N, D)

# -----------------------------
# 비디오(=프레임 폴더) 하나를 인코딩
# -----------------------------
def encode_one_video_frames(video_dir, model, preprocess, device, crop_type=0, batch_size=64):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    frame_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir)
                   if f.lower().endswith(exts)]
    if len(frame_paths) == 0:
        return None  # 프레임 없음

    frame_paths.sort(key=natural_key)

    feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i+batch_size]
            batch_tensors = []
            for p in batch_paths:
                img_bgr = cv2.imread(p)
                if img_bgr is None:
                    # 손상된 이미지 등은 스킵
                    continue
                cropped = image_crop(img_bgr, crop_type)                 # (224,224,3) RGB
                pil_img = Image.fromarray(cropped)                       # PIL RGB
                tensor = preprocess(pil_img).unsqueeze(0)                # (1,3,224,224)
                batch_tensors.append(tensor)

            if len(batch_tensors) == 0:
                continue

            batch = torch.cat(batch_tensors, dim=0).to(device)          # (B,3,224,224)
            feat = model.encode_image(batch)                             # (B,512) on ViT-B/16
            feat = feat.float().cpu()
            feats.append(feat)

    if len(feats) == 0:
        return None

    feats = torch.cat(feats, dim=0)                                     # (T,512)
    return feats.numpy()                                                # np.float32

# -----------------------------
# 루트 디렉토리 순회:
#   root/동영상1/동영상1/프레임들
#   root/동영상2/동영상2/프레임들
# 등의 "리프 폴더(프레임이 들어있는 폴더)"를 찾아 저장
# -----------------------------
def find_leaf_frame_dirs(root):
    leaf_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        # 이미지가 하나라도 있으면 '리프 프레임 폴더'로 간주
        if any(f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')) for f in filenames):
            leaf_dirs.append(dirpath)
    return sorted(leaf_dirs)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Encode frames to CLIP features and save as .npy")
    parser.add_argument('--root', type=str, default="../hackerton2-1/extracted_frames",
                        help='프레임들이 들어있는 루트 디렉토리 (예: /path/to/root)')
    parser.add_argument('--out', type=str, default="../hackerton2-1/CLIPFeatures",
                        help='출력 .npy를 저장할 루트 디렉토리')
    parser.add_argument('--crop_type', type=int, default=0, choices=list(range(10)),
                        help='크롭/플립 타입 (0~9, 올려준 코드와 동일)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='인퍼런스 배치 크기')
    parser.add_argument('--model', type=str, default="ViT-B/16",
                        help='CLIP 비전 백본 (예: ViT-B/16)')
    # --- 추가: 스니펫 풀링 옵션 ---
    parser.add_argument('--snippet_len', type=int, default=16,
                        help='스니펫 길이(프레임 수). 예: 16')
    parser.add_argument('--pool', type=str, default='mean', choices=['mean', 'max'],
                        help='스니펫 내 풀링 방식 (mean|max)')
    parser.add_argument('--remainder', type=str, default='drop', choices=['pad', 'drop'],
                        help="T가 snippet_len으로 나누어떨어지지 않을 때 처리 방식: pad(마지막 프레임 반복 패딩) 또는 drop(나머지 버림)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model, device=device)

    leaf_dirs = find_leaf_frame_dirs(args.root)
    if len(leaf_dirs) == 0:
        print(f"[경고] 이미지 프레임 폴더를 찾을 수 없습니다: {args.root}")
        return

    print(f"[정보] 인코딩 대상 폴더 수: {len(leaf_dirs)}")
    for leaf in tqdm(leaf_dirs, desc="Encoding videos"):
        rel = os.path.relpath(leaf, args.root)          # ex) "동영상1/동영상1"
        vid_name = os.path.basename(leaf)               # ex) "동영상1"
        out_dir = os.path.join(args.out, os.path.dirname(rel))
        os.makedirs(out_dir, exist_ok=True)

        # 원본 프레임 피처 저장 경로
        out_path = os.path.join(out_dir, f"{vid_name}.npy")
        # 스니펫 풀링 피처 저장 경로
        out_path_snip = os.path.join(
            out_dir, f"{vid_name}.npy"
        )

        feats = encode_one_video_frames(
            video_dir=leaf,
            model=model,
            preprocess=preprocess,
            device=device,
            crop_type=args.crop_type,
            batch_size=args.batch_size
        )

        if feats is None:
            print(f"[스킵] 프레임이 없거나 읽기 실패: {leaf}")
            continue

        # (T,512)
        np.save(out_path, feats.astype(np.float32))
        print("frame feats:", feats.shape)

        # (T,512) -> (T/snippet_len, 512)
        snip_feats = pool_snippets(
            feats=feats, snippet_len=args.snippet_len,
            pool=args.pool, remainder=args.remainder
        )
        np.save(out_path_snip, snip_feats.astype(np.float32))
        print("snippet feats:", snip_feats.shape, "->", out_path_snip)

    print("[완료] 모든 비디오 인코딩이 끝났습니다.")

if __name__ == "__main__":
    main()
