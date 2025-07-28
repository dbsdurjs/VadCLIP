import os
import json
import numpy as np
import argparse
from transformers import T5Tokenizer, T5EncoderModel
import torch
from tqdm import tqdm

def encode_and_save(caption_dict, tokenizer, model, batch_size, out_path, device):
    captions = list(caption_dict.values())
    if not captions:
        print(f"No captions in {out_path}, skipping.")
        return

    all_embeddings = []

    for i in range(0, len(captions), batch_size):
        batch = captions[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)  # T5EncoderModel
            last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, dim)

            # 평균 풀링
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)
            embeddings = (last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)  # (batch, dim)

        all_embeddings.append(embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)
    np.save(out_path, all_embeddings)
    print(f"Saved {out_path} (shape={all_embeddings.shape}), used {len(captions)} captions.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract caption features with T5 encoder')
    parser.add_argument("--caption-folder",      type=str, required=True)
    parser.add_argument("--caption-feat-folder", type=str, required=True)
    parser.add_argument("--batch-size",          type=int, default=256)
    parser.add_argument("--model-name",          type=str, default="t5-base")
    args = parser.parse_args()

    os.makedirs(args.caption_feat_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5EncoderModel.from_pretrained(args.model_name).to(device)
    model.eval()

    for fname in sorted(os.listdir(args.caption_folder)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(args.caption_folder, fname), 'r', encoding='utf-8') as f:
            caption_dict = json.load(f)

        out_fname = fname.replace(".json", ".npy")
        out_path  = os.path.join(args.caption_feat_folder, out_fname)

        encode_and_save(
            caption_dict,
            tokenizer,
            model,
            args.batch_size,
            out_path,
            device
        )
