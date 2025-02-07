# 16개 클래스 중 이전 8개 클래스 작업
import os
import torch
import multiprocessing
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm

# ✅ CPU 코어 개수 확인 후 적절한 num_workers 설정
NUM_WORKERS = 8  # 예시로 8개 사용
print(f"🔹 Using num_workers={NUM_WORKERS}")

# 📌 3. 모델 및 processor 로드 (먼저 로드하여 데이터셋에 전달)
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b")
model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip2-opt-6.7b", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 📌 1. 개별 프레임을 로딩하는 데이터셋 (processor를 통한 전처리 적용)
class FrameDataset(Dataset):
    def __init__(self, base_folder, processor):
        self.data = []
        self.video_names = []
        self.processor = processor  # processor를 멤버 변수로 저장

        classes_names = os.listdir(base_folder)
        classes_names = classes_names[8:][3]   # 8개 중 이전 4개만
        print(f'작업 폴더 이름 : {classes_names}')

        for class_name in os.listdir(base_folder):

            if class_name not in classes_names:
                continue
            
            class_path = os.path.join(base_folder, class_name)
            if not os.path.isdir(class_path):
                continue

            # 클래스 폴더 내부의 모든 동영상 폴더 가져오기
            video_folders = sorted(os.listdir(class_path))
            for video_folder in video_folders:
                video_folder_path = os.path.join(class_path, video_folder)
                if not os.path.isdir(video_folder_path):
                    continue

                # 선택된 동영상 폴더의 프레임 파일 리스트
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

        # 이미지 로드 및 RGB 변환
        image = Image.open(image_path).convert("RGB")
        # processor를 사용하여 전처리: 내부적으로 리사이즈, 정규화 등 적용됨
        inputs = self.processor(images=image, return_tensors="pt")
        # inputs['pixel_values']의 shape는 [1, C, H, W]이므로 squeeze로 배치 차원 제거
        pixel_values = inputs["pixel_values"].squeeze(0)
        
        return video_folder_path, image_path, pixel_values, video_name

# 📌 2. 데이터셋 및 데이터 로더 생성
base_folder = "/media/vcl/DATA/YG/Extracted_Frames/"
dataset = FrameDataset(base_folder, processor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

# 📌 4. 배치 단위로 캡션 생성 (비디오명 포함)
def generate_captions(dataloader, model, processor, device):
    model.eval()
    current_video = None

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Processing Videos", unit="batch", dynamic_ncols=True)

        for batch in progress_bar:
            video_folder_paths, image_paths, pixel_values, video_names = batch

            if current_video is None or current_video != video_names[0]:
                current_video = video_names[0]
                print(f"\n🎥 Processing video: {current_video}")

            pixel_values = pixel_values.to(device, torch.float16)

            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            captions = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for video_folder_path, image_path, caption in zip(video_folder_paths, image_paths, captions):
                frame_number = os.path.basename(image_path)
                output_file = os.path.join(video_folder_path, f"{os.path.basename(video_folder_path)}.txt")

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"{frame_number}: {caption}")

                # print(f'🖼️ Frame: {frame_number} | 📜 Caption: {caption}')

            progress_bar.set_postfix({"Current Video": current_video, "Processed Frames": len(pixel_values)})

        print("\n✅ All videos processed successfully!")

# 📌 4. 기존 파일 삭제 (이전 결과 지우기)
def delete_existing_files(base_folder):
    
    classes_names = os.listdir(base_folder)[8:][3] # 8개 중 이전 4개만
    print(f"삭제할 작업 폴더: {classes_names}")
    
    for class_name in classes_names:
        class_path = os.path.join(base_folder, class_name)
        if not os.path.isdir(class_path):
            continue
        
        for video_folder in os.listdir(class_path):
            video_folder_path = os.path.join(class_path, video_folder)
            if not os.path.isdir(video_folder_path):
                continue
            
            output_file = os.path.join(video_folder_path, f"{video_folder}.txt")
            if os.path.exists(output_file):
                os.remove(output_file)
                print(f"🗑️ Deleted existing file: {output_file}")

if __name__ == '__main__':

    # 📌 5. 캡션 생성 실행
    delete_existing_files(base_folder)
    generate_captions(dataloader, model, processor, device)
