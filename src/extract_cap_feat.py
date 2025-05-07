import os
import json
import torch
import numpy as np
from clip import clip
import argparse

CUDA_VISIBLE_DEVICES=0
def build_short_caption(cap: str, device: str, max_tokens: int = 77) -> str:
    """
    1) 온점(.)으로 문장 단위 분리
    2) 각 문장을 tokenize(truncate=True)해서 실제 토큰 수 계산
    3) 누적 토큰 수가 max_tokens 미만인 문장만 모아서 반환
    """
    sentences = [s.strip() for s in cap.split('.') if s.strip()]
    kept, cum_tokens = [], 0

    for sent in sentences:
        # 토큰 수만 계산하기 위해 truncate=True
        tokens = clip.tokenize([sent], truncate=True).to(device)  # (1,77)
        tok_count = (tokens != 0).sum().item()                    # 실제 토큰 수

        # 문장 하나가 max_tokens 이상이면 아예 건너뛰기
        if tok_count >= max_tokens:
            continue

        # 누적토큰 + 이 문장 토큰 < max_tokens 인 경우에만 추가
        if cum_tokens + tok_count < max_tokens:
            kept.append(sent)
            cum_tokens += tok_count
        else:
            break

    # 문장들 사이에 온점+공백을 넣어 재결합
    return '. '.join(kept)

def encode_and_save(caption_dict, model, device, batch_size, out_path):
    # 1) 각 캡션을 문장 단위로 잘라 새 캡션 생성
    processed_caps = []
    for frame_name, cap in caption_dict.items():
        short_cap = build_short_caption(cap, device)
        if short_cap:  # 잘린 결과가 비어있지 않으면 사용
            processed_caps.append(short_cap)

    if not processed_caps:
        print(f"No valid captions in {out_path}, skipping.")
        return

    # 2) 배치 단위로 tokenize → encode (truncate=False)
    all_feats = []
    for i in range(0, len(processed_caps), batch_size):
        batch = processed_caps[i : i+batch_size]
        tokens = clip.tokenize(batch).to(device)                 # (B, max_len) 
        with torch.no_grad():
            feats = model.encode_text_cap(tokens)               # (B, 512)
        all_feats.append(feats.float().cpu().numpy())
        torch.cuda.empty_cache()

    # 3) 결과 합치고 저장
    all_feats = np.concatenate(all_feats, axis=0)               # (num_valid,512)
    np.save(out_path, all_feats)
    print(f"Saved {out_path} (shape={all_feats.shape}), used {len(processed_caps)}/{len(caption_dict)} captions.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-folder",      type=str)
    parser.add_argument("--caption-feat-folder", type=str)
    parser.add_argument("--batch-size",         type=int, default=1024)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/16", device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    os.makedirs(args.caption_feat_folder, exist_ok=True)

    for fname in sorted(os.listdir(args.caption_folder)):
        if not fname.endswith(".json"):
            continue
        json_path = os.path.join(args.caption_folder, fname)
        with open(json_path, 'r', encoding='utf-8') as f:
            caption_dict = json.load(f)

        out_fname = fname.replace(".json", ".npy")
        out_path  = os.path.join(args.caption_feat_folder, out_fname)

        encode_and_save(
            caption_dict,
            model,
            device,
            batch_size=args.batch_size,
            out_path=out_path
        )
