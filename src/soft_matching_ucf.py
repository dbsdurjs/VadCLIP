import os, csv
import numpy as np
import torch
from typing import Dict, Optional

IMAGE_CSV_PATH   = 'list/ucf_CLIP_rgb.csv'
CAPTION_CSV_PATH = 'list/ucf_CLIP_rgb_description.csv'
SAVE_DIR         = '../VAD_dataset/UCF-Crimes/UCF_Crimes/refine_description/ucf_output_refine_1'
SAVE_DIR_META         = '../VAD_dataset/UCF-Crimes/UCF_Crimes/refine_description/ucf_output/ucf_meta_refine_1'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR_META, exist_ok=True)

def padding_(img_emb, text_emb):
    if text_emb.ndim > 1 and text_emb.shape[0] > 1:
        print(f"Text embedding shape: {text_emb.shape}")
        text_emb = text_emb / torch.norm(text_emb, dim=-1, keepdim=True)
    else:
        print(f"Unexpected text embedding shape: {text_emb.shape}")
        text_emb = text_emb / torch.norm(text_emb, dim=-1, keepdim=True)

    if img_emb.ndim > 1 and img_emb.shape[0] > 1:
        print(f"Image embedding shape: {img_emb.shape}")
        img_emb = img_emb / torch.norm(img_emb, dim=-1, keepdim=True)
    else:
        print(f"Unexpected image embedding shape: {img_emb.shape}")
        img_emb = img_emb / torch.norm(img_emb, dim=-1, keepdim=True)
    
    # 형상 불일치 처리
    if img_emb.ndim > 1 and text_emb.ndim > 1 and img_emb.shape[0] != text_emb.shape[0]:
        print(f"Shape mismatch: img_emb {img_emb.shape}, text_emb {text_emb.shape}")
        if img_emb.shape[0] > text_emb.shape[0]:
            # 텍스트 임베딩이 더 짧으면 마지막 벡터 복사
            diff = img_emb.shape[0] - text_emb.shape[0]
            last_text_emb = text_emb[-1:].repeat(diff, 1)  # 마지막 벡터 복사
            text_emb = torch.cat([text_emb, last_text_emb], dim=0)
            print(f"Adjusted text_emb shape to: {text_emb.shape}")

    return img_emb, text_emb

def key_upto_x264(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]  # ex) Abuse001_x264__0
    if "_x264" in stem:
        idx = stem.index("_x264")
        return stem[:idx+5]  # include "_x264"
    # fallback: '__' 이전까지
    if "__" in stem:
        return stem.split("__")[0]
    # 또다른 fallback: '_captions' 또는 '_description' 앞부분
    for tok in ["_captions", "_description"]:
        if tok in stem:
            return stem.split(tok)[0]
    return stem

def stem_without_ext(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def read_paths(csv_path: str) -> list:
    items = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if 'path' not in reader.fieldnames:
            raise ValueError(f"[ERR] CSV '{csv_path}'에 'path' 컬럼이 없습니다. 필드: {reader.fieldnames}")
        for row in reader:
            items.append(row['path'])
    return items

def npy_to_torch(path: str, device='cuda') -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 512:
        raise ValueError(f"[ERR] {path} shape={arr.shape}, 기대=(*,512)")
    return torch.from_numpy(arr.astype(np.float32)).to(device)

# =========================================
# 코사인/정렬 유틸
# =========================================
def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True))

@torch.no_grad()
def hard_align_scores(
    img_emb: torch.Tensor,   # (N, 512)
    txt_emb: torch.Tensor    # (M, 512)
):
    """
    compute_clip_score의 1:1 매핑 로직을 반영한 하드 매칭 점수 계산:
    - 이미지/텍스트 임베딩 L2 정규화
    - N != M이고 N > M이면 텍스트 마지막 벡터를 반복해 (N,512)로 패딩
    - 1:1 코사인 유사도 -> 프레임별 점수, 비디오 평균
    - j_idx(텍스트 인덱스 매핑)과 t_hard(하드 대응 텍스트 임베딩) 반환
    """
    device = img_emb.device
    dtype  = img_emb.dtype

    # float32 보정(안전)
    if img_emb.dtype != torch.float32:
        img_emb = img_emb.float()
    if txt_emb.dtype != torch.float32:
        txt_emb = txt_emb.float()

    # L2 정규화
    img = img_emb / (img_emb.norm(dim=-1, keepdim=True))
    txt = txt_emb / (txt_emb.norm(dim=-1, keepdim=True))

    N = img.shape[0]
    M = txt.shape[0]

    # --- 길이 불일치 처리 (compute_clip_score와 동일 아이디어) ---
    if N != M and N > M:
        # 텍스트가 더 짧으면 마지막 행을 반복해서 패딩
        diff = N - M
        last_txt = txt[-1:].repeat(diff, 1)          # (diff, 512)
        txt = torch.cat([txt, last_txt], dim=0)      # (N, 512)
        M = txt.shape[0]                              # 이제 M == N

    # j_idx 구성: 1:1 매핑에 대응하는 텍스트 인덱스(패딩 이후에도 필요)
    # - 기본은 선형 매핑(레이트 보정): round((M-1)/(N-1) * i)
    # - 위에서 N>M일 땐 패딩으로 M==N이 되었으므로 j_idx == [0..N-1]
    if N == 1:
        j_idx = torch.zeros(1, dtype=torch.long, device=device)
    else:
        alpha = (M - 1) / max(1, (N - 1))
        j_idx = torch.round(torch.arange(N, device=device) * alpha).long().clamp(0, M - 1)

    # 선택 텍스트 임베딩(1:1)
    sel_txt = txt[j_idx]  # (N, 512)

    # 코사인 유사도(프레임별)
    frame_scores = torch.nn.functional.cosine_similarity(img, sel_txt, dim=-1)  # (N,)
    video_score  = frame_scores.mean()

    # 하드 대응 텍스트 임베딩(t_hard)도 함께 반환(후속 단계에서 사용)
    t_hard = sel_txt

    return {
        "frame_scores": frame_scores.to(device=device, dtype=dtype),  # (N,)
        "video_score":  video_score.to(device=device, dtype=dtype),   # scalar
        "j_idx":        j_idx,                                        # (N,)
        "t_hard":       t_hard.to(device=device, dtype=dtype),        # (N,512)
    }


@torch.no_grad()
def soft_match_text_embeddings(
    img_emb: torch.Tensor,         # (N,512)
    txt_emb: torch.Tensor,         # (M,512)
    temperature: float = 0.06,
    topk: int = 10,
    band_window: Optional[int] = 12,      # 시간 밴드 폭(±W)
    time_prior_sigma: Optional[float] = 6.0,
    chunk_size: int = 512
):
    device = img_emb.device
    img = _l2norm(img_emb)
    txt = _l2norm(txt_emb)
    N, D = img.shape
    M, _ = txt.shape

    alpha = (M - 1) / max(1, (N - 1))
    out_ttilde = torch.empty((N, D), device=device, dtype=img.dtype)
    frame_scores = torch.empty((N,), device=device, dtype=img.dtype)

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        img_chunk = img[start:end]      # (C,512)
        C = end - start
        i_idx = torch.arange(start, end, device=device).float()
        j_center = (alpha * i_idx).round().long().clamp(0, M - 1)  # (C,)

        for c in range(C):
            jc = int(j_center[c].item())
            if band_window is None:
                j0, j1 = 0, M
            else:
                j0 = max(0, jc - band_window)
                j1 = min(M, jc + band_window + 1)

            sim = (img_chunk[c:c+1] @ txt[j0:j1].t()).squeeze(0)   # (B,)
            if sim.numel() == 0:
                out_ttilde[start + c] = torch.zeros(D, device=device, dtype=img.dtype)
                frame_scores[start + c] = torch.tensor(0., device=device, dtype=img.dtype)
                continue

            k = min(topk, sim.shape[0])
            topv, ridx = torch.topk(sim, k=k, dim=0)               # (k,)
            j_sel = (ridx + j0)

            if time_prior_sigma is not None:
                prior = -((j_sel.float() - j_center[c].float())**2) / (2 * (time_prior_sigma**2))
                logits = topv / temperature + prior
            else:
                logits = topv / temperature

            w = torch.softmax(logits, dim=0)                       # (k,)
            tsel = txt[j_sel]                                      # (k,512)
            ttilde = (w.unsqueeze(1) * tsel).sum(dim=0)            # (512,)
            out_ttilde[start + c] = ttilde
            frame_scores[start + c] = (img_chunk[c] * ttilde).sum()

    video_score = frame_scores.mean()
    return {"t_tilde": out_ttilde, "frame_scores": frame_scores, "video_score": video_score}

@torch.no_grad()
def apply_soft_matching_if_needed(
    img_emb: torch.Tensor,                    # (N,512)
    txt_emb: torch.Tensor,                    # (M,512)
    global_floor: float = 0.29,               # 기존 평균 clip score
    mad_coeff: float = 0.5,
    temperature: float = 0.06,                # 얼마나 많은 캡션을 섞을지
    topk: int = 16,                           # 후보군 캡션 내에서 사용할 캡션 개수(이미지에서 사용하는 16 사용)
    band_window: int = 8,                     # 후보군 캡션 범위 정하기(+-band window)
    time_prior_sigma: float = 6.0,            # 중앙에 있는 캡션일수록 가중치 증가(유사도가 높다고 가중치를 과도하게 주는 일 방지)
    min_improve: float = 0.02,                # soft matching 사용 시 최소 성능 향상량
):
    hard = hard_align_scores(img_emb, txt_emb)
    hard_vs = float(hard["video_score"].item())
    fs = hard["frame_scores"].detach().cpu().numpy()
    med = float(np.median(fs)); mad = float(np.median(np.abs(fs - med)) + 1e-8)
    T = max(global_floor, med - mad_coeff * mad)

    soft = soft_match_text_embeddings(
        img_emb, txt_emb,
        temperature=temperature, topk=topk,
        band_window=band_window, time_prior_sigma=time_prior_sigma
    )
    soft_vs = float(soft["video_score"].item())

    use_soft = ((soft_vs - hard_vs) >= min_improve) or (hard_vs < T and soft_vs >= T)

    if use_soft:
        return {
            "mode": "soft",
            "video_score": soft["video_score"],
            "frame_scores": soft["frame_scores"],
            "t_tilde": soft["t_tilde"],          # (N,512)
            "threshold_T": torch.tensor(T, device=img_emb.device),
            "hard_video_score": torch.tensor(hard_vs, device=img_emb.device),
        }
    else:
        txt = _l2norm(txt_emb)
        t_hard = txt[hard["j_idx"]]
        return {
            "mode": "hard",
            "video_score": hard["video_score"],
            "frame_scores": hard["frame_scores"],
            "t_tilde": t_hard,                   # (N,512)
            "threshold_T": torch.tensor(T, device=img_emb.device),
            "soft_video_score": torch.tensor(soft_vs, device=img_emb.device),
        }

# =========================================
# 메인: 키를 "_x264"까지 맞춰 매칭
#   - 이미지: 같은 키를 가진 모든 파일(__0,__1...) 각각 처리
#   - 캡션: 같은 키를 가진 대표 1개(또는 여러 개면 첫 번째) 사용
# =========================================
def run_softmatch_with_x264_key(
    image_csv_path=IMAGE_CSV_PATH,
    caption_csv_path=CAPTION_CSV_PATH,
    device='cuda',
    save_dir=SAVE_DIR,
    save_dir_meta=SAVE_DIR_META,
    temperature=0.06, topk=10, band_window=12, time_prior_sigma=6.0,
    global_floor=0.29, mad_coeff=0.5, min_improve=0.02,
):
    # 1) CSV 읽기
    img_paths = read_paths(image_csv_path)
    cap_paths = read_paths(caption_csv_path)

    # 2) 캡션: 키 -> 첫 경로 (여러 개면 첫 것)
    cap_by_key: Dict[str, str] = {}
    for p in cap_paths:
        k = key_upto_x264(p)
        cap_by_key.setdefault(k, p)  # 같은 키가 여러 개면 처음 것 사용(필요시 정책 변경)

    # 3) 이미지: 각 파일 별로 키 산출 후, 해당 키의 캡션과 1:1 처리
    matched = 0
    for img_p in img_paths:
        # if img_p == '../VAD_dataset/UCFClipFeatures/Burglary/Burglary095_x264__1.npy':
        #     breakpoint()
        key = key_upto_x264(img_p)
        cap_p = cap_by_key.get(key, None)
        if cap_p is None:
            print(f"[SKIP] 캡션 없음(key={key}): {img_p}")
            continue
        if not (os.path.isfile(img_p) and os.path.isfile(cap_p)):
            print(f"[WARN] 파일 누락: {img_p} or {cap_p}")
            continue

        try:
            img_emb = npy_to_torch(img_p, device=device)  # (N,512)
            txt_emb = npy_to_torch(cap_p, device=device)  # (M,512)

            img_emb, txt_emb = padding_(img_emb, txt_emb)

            out = apply_soft_matching_if_needed(
                img_emb, txt_emb,
                global_floor=global_floor, mad_coeff=mad_coeff,
                temperature=temperature, topk=topk,
                band_window=band_window, time_prior_sigma=time_prior_sigma,
                min_improve=min_improve,
            )

            # 저장: 이미지 파일 스템(전처리 구분 포함) 기준
            stem_img = stem_without_ext(img_p)  # ex) Abuse001_x264__0
            np.save(os.path.join(save_dir, f"{stem_img}_refine_1.npy"), # 1번 방법 - 평균 clip score 미만 soft match(_refine_1), 2번 방법 - 모두 soft match(_refine_2)
                    out["t_tilde"].detach().cpu().numpy())
            with open(os.path.join(save_dir_meta, f"{stem_img}__meta.txt"), "w") as f:
                f.write(f"mode={out['mode']}\n")
                f.write(f"video_score={float(out['video_score'].item()):.6f}\n")
                if 'soft_video_score' in out:
                    f.write(f"soft_video_score={float(out['soft_video_score'].item()):.6f}\n")
                if 'hard_video_score' in out:
                    f.write(f"hard_video_score={float(out['hard_video_score'].item()):.6f}\n")
                f.write(f"threshold_T={float(out['threshold_T'].item()):.6f}\n")
                f.write(f"image_path={img_p}\n")
                f.write(f"caption_path={cap_p}\n")
                f.write(f"match_key={key}\n")

            matched += 1
            print(f"[OK] {stem_img}  <=  {os.path.basename(cap_p)}   (key={key})   mode={out['mode']}")

        except Exception as e:
            print(f"[ERR] {img_p}: {e}")

    print(f"[DONE] matched pairs processed: {matched}")

if __name__ == "__main__": # clip score 기준으로 soft match 진행
    run_softmatch_with_x264_key(
        image_csv_path=IMAGE_CSV_PATH,
        caption_csv_path=CAPTION_CSV_PATH,
        device='cuda',  # 필요시 'cpu', cuda 실행 시 에러 발생
        save_dir=SAVE_DIR,
        save_dir_meta=SAVE_DIR_META,
        temperature=0.06, topk=16, band_window=8, time_prior_sigma=4.0,
        global_floor=0.29, mad_coeff=0.5, min_improve=0.02,
    )
