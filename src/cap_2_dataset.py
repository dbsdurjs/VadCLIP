import os
import torch
import multiprocessing
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm

# ✅ CPU 코어 개수 확인 후 적절한 num_workers 설정
NUM_WORKERS = multiprocessing.cpu_count()  # CPU 코어 절반 사용
print(f"🔹 Using num_workers={NUM_WORKERS}")

# 📌 1. 개별 프레임을 로딩하는 데이터셋 (각 클래스의 앞 절반만 선택)
class FrameDataset(Dataset):
    def __init__(self, base_folder):
        self.data = []
        self.video_names = []
        self.transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert("RGB")),  # ✅ 멀티스레딩 최적화
            transforms.ToTensor(),
        ])

        for class_name in os.listdir(base_folder):
            classes_names = os.listdir(base_folder)
            classes_names = classes_names[:8]   # 8개 작업, 서버에서 바꾸기
            print(f'작업 폴더 이름 : {classes_names}')

            if class_name not in classes_names:
                continue
            
            class_path = os.path.join(base_folder, class_name)
            if not os.path.isdir(class_path):
                continue

            # 🔹 클래스 폴더 내부의 모든 동영상 폴더 가져오기
            # video_folders = sorted(os.listdir(class_path))[:10]
            video_folders = sorted(os.listdir(class_path))
            for video_folder in video_folders:
                video_folder_path = os.path.join(class_path, video_folder)
                if not os.path.isdir(video_folder_path):
                    continue

                # 📌 선택된 동영상 폴더의 프레임 파일 리스트
                image_files = sorted([
                    os.path.join(video_folder_path, f) for f in os.listdir(video_folder_path)
                    if f.lower().endswith('.jpg')
                ])
                
                for img in image_files:
                    self.data.append((video_folder_path, img))
                    self.video_names.append(video_folder)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_folder_path, image_path = self.data[index]
        video_name = self.video_names[index]

        # ✅ `PIL` 대신 `transforms.Lambda()` 사용하여 멀티스레딩 활용
        image = Image.open(image_path)
        image = self.transform(image)
        
        return video_folder_path, image_path, image, video_name

# 📌 2. 데이터셋 및 데이터 로더 생성
# base_folder = "/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Extracted_Frames/"
base_folder= "/media/vcl/DATA/YG/Extracted_Frames/"

dataset = FrameDataset(base_folder)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

# 📌 3. 모델 로드
# processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
# model = AutoModelForImageTextToText.from_pretrained("microsoft/git-large-coco", torch_dtype=torch.float16)

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b")
model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip2-opt-6.7b", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 📌 4. 배치 단위로 캡션 생성 (비디오명 포함)
def generate_captions(dataloader, model, processor, device):
    model.eval()
    current_video = None

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Processing Videos", unit="batch", dynamic_ncols=True)

        for batch in progress_bar:
            video_folder_paths, image_paths, images, video_names = batch

            if current_video is None or current_video != video_names[0]:
                current_video = video_names[0]
                print(f"\n🎥 Processing video: {current_video}")

            pixel_values = images.to(device, torch.float16)

            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            captions = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for video_folder_path, image_path, caption in zip(video_folder_paths, image_paths, captions):
                frame_number = os.path.basename(image_path)
                output_file = os.path.join(video_folder_path, f"{os.path.basename(video_folder_path)}.txt")

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"{frame_number}: {caption}\n")

                print(f'🖼️ Frame: {frame_number} | 📜 Caption: {caption}')

            progress_bar.set_postfix({"Current Video": current_video, "Processed Frames": len(images)})

        print("\n✅ All videos processed successfully!")

# 📌 4. 기존 파일 삭제 (이전 결과 지우기)
def delete_existing_files(base_folder):
    for class_name in os.listdir(base_folder):
        class_path = os.path.join(base_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        for video_folder in os.listdir(class_path):
            video_folder_path = os.path.join(class_path, video_folder)
            if not os.path.isdir(video_folder_path):
                continue

            output_file = os.path.join(video_folder_path, f"{video_folder}.txt")

            # ✅ 기존 파일 삭제
            if os.path.exists(output_file):
                os.remove(output_file)
                print(f"🗑️ Deleted existing file: {output_file}")

# 📌 5. 캡션 생성 실행
delete_existing_files(base_folder)
generate_captions(dataloader, model, processor, device)
