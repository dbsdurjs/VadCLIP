import os
import json
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer

def encode_and_save(caption_dict, model, batch_size, out_path):
    captions = list(caption_dict.values())
    if not captions:
        print(f"No captions in {out_path}, skipping.")
        return

    # SBERT가 내부에서 토크나이즈, 패딩, 트렁케이트를 처리
    embeddings = model.encode(
        captions,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True
    )  # (num_captions, embedding_dim)

    np.save(out_path, embeddings.astype(np.float32))
    print(f"Saved {out_path} (shape={embeddings.shape}), used {len(captions)} captions.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract caption features with text encoder')
    parser.add_argument("--caption-folder",      type=str, required=True)
    parser.add_argument("--caption-feat-folder", type=str, required=True)
    parser.add_argument("--batch-size",         type=int, default=256)
    parser.add_argument("--model-name",         type=str,
                        default="sentence-transformers/all-mpnet-base-v2")
    args = parser.parse_args()

    os.makedirs(args.caption_feat_folder, exist_ok=True)
    model = SentenceTransformer(args.model_name)

    for fname in sorted(os.listdir(args.caption_folder)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(args.caption_folder, fname), 'r', encoding='utf-8') as f:
            caption_dict = json.load(f)

        out_fname = fname.replace(".json", ".npy")
        out_path  = os.path.join(args.caption_feat_folder, out_fname)

        encode_and_save(
            caption_dict,
            model,
            batch_size=args.batch_size,
            out_path=out_path
        )