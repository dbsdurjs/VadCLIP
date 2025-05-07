import os
import json
import argparse
import torch
from PIL import Image
import re
from transformers import AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from janus.models import MultiModalityCausalLM, VLChatProcessor
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ImageFolderDataset(Dataset):
    def __init__(self, image_paths):
        self.paths = image_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return img, path

def collate_fn(batch):
    images, paths = zip(*batch)
    return list(images), list(paths)

def generate_captions_nested(
    base_image_folder: str,
    model_path: str,
    batch_size: int,
    mapping: dict,
    max_tokens: int = 512,
    num_workers: int = 8
):
    # 모델 및 프로세서 로드 (한 번만)
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    # 출력 디렉토리 생성
    output_root = os.path.join(os.path.dirname(base_image_folder), "Extracted_Captions_janus")
    os.makedirs(output_root, exist_ok=True)

    class_list = list(mapping.values())
    class_options = ", ".join(class_list)

    # 하위 디렉토리(동영상 폴더) 순회
    for video_dir in sorted(os.listdir(base_image_folder)):
        video_path = os.path.join(base_image_folder, video_dir)
        if not os.path.isdir(video_path):
            continue

        # 16프레임마다 이미지 선택
        image_files = []
        for fn in sorted(os.listdir(video_path)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            frame_match = re.search(r"_frame_(\d+)", fn)
            if frame_match and (int(frame_match.group(1))+1) % 16 == 0:
                image_files.append(os.path.join(video_path, fn))
        if not image_files:
            print(f"Skipping {video_dir}: No images at 16-frame intervals")
            continue

        enhanced_prompt = (
            f"First, check whether this scene belongs to or resembles any of the following categories: "
            f"{class_options}. "
            "If so, name the matching category and describe the core action with a vivid, precise verb, "
            "include a brief detail of the setting, and state the cause or purpose to show why it fits. "
            "Otherwise, briefly describe the main elements. "
            "Use present tense and factual language based only on what is visible. "
            "Ensure your caption ends with a complete sentence."
        )

        # DataLoader 설정
        dataset = ImageFolderDataset(image_files)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=False
        )

        captions = {}
        print(f"input prompt : {enhanced_prompt}")
        print(f"Processing folder: {video_dir}, {len(image_files)} images)")

        # 배치 처리
        with torch.no_grad():  # 그래디언트 계산 비활성화
            for pil_images, batch_paths in tqdm(loader, desc=f"Processing {video_dir}", leave=False):
                # 대화 컨텍스트 생성
                convs = [
                    [
                        {"role": "<|User|>", "content": f"<image_placeholder>\n{enhanced_prompt}", "images": [path]},
                        {"role": "<|Assistant|>", "content": ""}
                    ]
                    for path in batch_paths
                ]

                # 개별 프로세싱 후 배치화
                prepares = [vl_chat_processor(conversations=conv, images=[img], force_batchify=False)
                           for conv, img in zip(convs, pil_images)]
                prepare_inputs = vl_chat_processor.batchify(prepares).to(vl_gpt.device)

                # 캡션 생성
                inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                outputs = vl_gpt.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    use_cache=True
                )

                # 출력 검증
                if outputs.shape[0] != len(batch_paths):
                    print(f"Warning: Batch size mismatch (expected {len(batch_paths)}, got {outputs.shape[0]})")

                # 디코딩 및 저장
                for img_path, output_ids in zip(batch_paths, outputs):
                    caption = tokenizer.decode(output_ids.cpu().tolist(), skip_special_tokens=True)
                    img_name = os.path.basename(img_path)
                    captions[img_name] = caption

        # JSON 저장
        out_json = os.path.join(output_root, f"{video_dir}_captions_janus_pro.json")
        with open(out_json, 'w', encoding='utf-8') as jf:
            json.dump(captions, jf, ensure_ascii=False, indent=2)
        print(f"Saved captions for '{video_dir}' to {out_json}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract captions from nested image folders using Janus-Pro-7B"
    )
    parser.add_argument(
        "--img-folder", "-i", default="../VAD_dataset/XD-Violence/Extracted_Frames",
        help="기준이 되는 최상위 이미지 폴더 경로"
    )
    parser.add_argument(
        "--model-path", "-m", default="deepseek-ai/Janus-Pro-7B",
        help="HuggingFace에 등록된 모델 경로"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=8,
        help="한 배치에 처리할 이미지 개수"
    )
    parser.add_argument(
        "--max-tokens", "-t", type=int, default=150,
        help="최대 생성 토큰 수"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8,
        help="DataLoader 워커 수"
    )
    args = parser.parse_args()

    mapping = {
        'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot',
        'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'
    }

    generate_captions_nested(
        base_image_folder=args.img_folder,
        model_path=args.model_path,
        batch_size=args.batch_size,
        mapping=mapping,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers
    )