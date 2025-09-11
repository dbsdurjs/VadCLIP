import pandas as pd
import numpy as np
import torch
import os
from collections import defaultdict

def get_video_name(path):
    base = os.path.basename(path)
    if '__' in base:
        return base.split('__')[0]  # Abuse001_x264__0.npy -> Abuse001_x264
    elif '_captions' in base:
        return '_'.join(base.split('_')[:-3]) if 'janus_pro' in base else '_'.join(base.split('_')[:-3])
    return base.replace('.npy', '')

def compute_clip_score(image_embs_list, text_emb):
    scores = []
    individual_scores = []  # 개별 유사도 저장
    text_emb = torch.from_numpy(text_emb).float()
    
    # 텍스트 임베딩 형상 확인
    if text_emb.ndim > 1 and text_emb.shape[0] > 1:
        print(f"Text embedding shape: {text_emb.shape}")
        text_emb = text_emb / torch.norm(text_emb, dim=-1, keepdim=True)  # (170, 512)
    else:
        print(f"Unexpected text embedding shape: {text_emb.shape}")
        text_emb = text_emb / torch.norm(text_emb, dim=-1, keepdim=True)
    
    for img_emb in image_embs_list:
        img_emb = torch.from_numpy(img_emb).float()
        if img_emb.ndim > 1 and img_emb.shape[0] > 1:
            print(f"Image embedding shape: {img_emb.shape}")
            img_emb = img_emb / torch.norm(img_emb, dim=-1, keepdim=True)  # (170, 512)
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
                text_emb = torch.cat([text_emb, last_text_emb], dim=0)  # (11301, 512)
                print(f"Adjusted text_emb shape to: {text_emb.shape}")
        
        # 1:1 매핑으로 코사인 유사도 계산
        if img_emb.shape == text_emb.shape and img_emb.ndim > 1:
            similarities = torch.nn.functional.cosine_similarity(img_emb, text_emb, dim=-1)  # (11301,)
            individual_scores.extend(similarities.tolist())  # 1:1 매핑 유사도 저장
            score = similarities.mean().item()  # 평균 유사도
        else:
            print(f"Shape mismatch after adjustment: img_emb {img_emb.shape}, text_emb {text_emb.shape}")
            score = torch.nn.functional.cosine_similarity(img_emb, text_emb, dim=-1).item()
            individual_scores.append(score)
        scores.append(score)
    
    return np.mean(scores), individual_scores

def main(output_dir):
    # CSV 파일 경로
    image_csv_path = 'list/ucf_CLIP_rgb.csv'  # 실제 경로로 변경
    caption_csv_path = 'list/ucf_CLIP_rgb_description_refine_2.csv'  # 실제 경로로 변경

    # CSV 파일 로드
    try:
        df_image = pd.read_csv(image_csv_path)
        df_caption = pd.read_csv(caption_csv_path)
    except FileNotFoundError as e:
        print(f"Error: CSV file not found: {e}")
        return

    # 이미지 임베딩을 비디오 이름으로 그룹핑
    image_groups = defaultdict(list)
    for _, row in df_image.iterrows():
        video_name = get_video_name(row['path'])
        image_groups[video_name].append(row['path'])

    # 캡션 임베딩 매핑
    caption_map = {}
    for _, row in df_caption.iterrows():
        video_name = get_video_name(row['path'])
        caption_map[video_name] = row['path']

    # 각 비디오의 CLIPScore 및 개별 유사도 저장
    video_scores = {}
    all_scores = []
    video_individual_scores = {}  # 비디오별 개별 유사도 저장

    for video_name, image_paths in image_groups.items():
        if video_name not in caption_map:
            print(f"Warning: No caption for {video_name}")
            continue
        
        # 이미지 임베딩 로드
        image_embs = []
        for path in image_paths:
            if os.path.exists(path):
                try:
                    emb = np.load(path)
                    if emb.ndim > 1 and emb.shape[0] == 1:
                        emb = emb.squeeze(0)
                    print(f"Loaded image embedding {path} with shape {emb.shape}")
                    image_embs.append(emb)
                except Exception as e:
                    print(f"Error loading image embedding {path}: {e}")
            else:
                print(f"File not found: {path}")
        
        if not image_embs:
            print(f"No valid image embeddings for {video_name}")
            continue
        
        # 캡션 임베딩 로드
        caption_path = caption_map[video_name]
        if os.path.exists(caption_path):
            try:
                text_emb = np.load(caption_path)
                if text_emb.ndim > 1 and text_emb.shape[0] == 1:
                    text_emb = text_emb.squeeze(0)
                print(f"Loaded text embedding {caption_path} with shape {text_emb.shape}")
            except Exception as e:
                print(f"Error loading text embedding {caption_path}: {e}")
                continue
        else:
            print(f"File not found: {caption_path}")
            continue
        
        # CLIPScore 및 개별 유사도 계산
        try:
            score, ind_scores = compute_clip_score(image_embs, text_emb)
            video_scores[video_name] = score
            all_scores.append(score)
            video_individual_scores[video_name] = ind_scores
            print(f"{video_name} CLIPScore: {score:.4f}, Individual 1:1 scores: {len(ind_scores)} values")
        except Exception as e:
            print(f"Error computing CLIPScore for {video_name}: {e}")

    # 모든 파일(비디오)의 평균 CLIPScore 계산
    if all_scores:
        avg_score = np.mean(all_scores)
        print(f"Average CLIPScore across all videos: {avg_score:.4f}")
    else:
        print("No scores computed.")

    # 결과 저장
    try:
        # np.save('video_clip_scores.npy', video_scores)  # 딕셔너리 저장
        # np.save('average_clip_score.npy', np.array([avg_score]))  # 평균 저장
        pd.DataFrame.from_dict(video_scores, orient='index', columns=['CLIPScore']).to_csv(os.path.join(output_dir, 'video_clip_scores_ucf.csv'))
        pd.DataFrame({'Average_CLIPScore': [avg_score]}).to_csv(os.path.join(output_dir, 'average_clip_score_ucf.csv'), index=False)
        # 개별 유사도 저장
        # np.save('individual_clip_scores.npy', video_individual_scores)
        # CSV로 개별 유사도 저장
        # ind_scores_df = pd.DataFrame([(video, score) for video, scores in video_individual_scores.items() for score in scores], 
        #                              columns=['Video', 'Individual_CLIPScore'])
        # ind_scores_df.to_csv(os.path.join(output_dir, 'individual_clip_scores_ucf.csv'), index=False)
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    output_dir = '../clip_score/ucf/after_refine_2'
    os.makedirs(output_dir, exist_ok=True)
    main(output_dir)