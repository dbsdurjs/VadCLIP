import torch
from clip import clip

def initialize_vlm_model_and_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CLIP 모델 로드 (ViT-B/16 사용)
    model, preprocess = clip.load("ViT-B/16", device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model, preprocess, device