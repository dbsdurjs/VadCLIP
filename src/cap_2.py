# data loader를 사용 안 하는 코드, cap_dataset, cap_2_dataset이 메인
# 참고용
import os, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

if __name__ == '__main__':
    # folder = "/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Extracted_Frames/Arrest/Arrest001_x264"
    base_folder = "/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Extracted_Frames/"
    # base_folder= "/media/vcl/DATA/YG/Extracted_Frames/"
    
    # processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b")
    # model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip2-opt-6.7b", torch_dtype=torch.float16)

    processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    model = AutoModelForImageTextToText.from_pretrained("microsoft/git-large-coco", torch_dtype=torch.float16)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for class_name in os.listdir(base_folder):
        classes_names = os.listdir(base_folder)
        classes_names = classes_names[8:]   # 8개 작업, 서버에서 바꾸기
        print(f'작업 폴더 이름 : {classes_names}')

        if class_name not in classes_names:
            continue

        class_path = os.path.join(base_folder, class_name)
        if not os.path.isdir(class_path):  # 디렉토리가 아니면 건너뛰기
            continue

        # 4️⃣ 각 동영상 폴더 순회
        for video_folder in os.listdir(class_path):
            video_folder_path = os.path.join(class_path, video_folder)
            
            if not os.path.isdir(video_folder_path):  # 디렉토리가 아니면 건너뛰기
                continue

            print(f"Processing video frames in: {video_folder_path}")

            # 5️⃣ 동영상 폴더 내 이미지 파일 리스트 정렬
            image_files = sorted([
                os.path.join(video_folder_path, f) for f in os.listdir(video_folder_path)
                if f.lower().endswith('.jpg')
            ])

            if not image_files:
                print(f"❌ No image frames found in {video_folder_path}")
                continue

            # 6️⃣ 텍스트 파일 설정 (동영상 폴더명 기반으로 저장)
            output_file = os.path.join(video_folder_path, f"{video_folder}.txt")

            with open(output_file, "w", encoding="utf-8") as f:
                # 7️⃣ 각 프레임에 대해 캡션 생성 및 저장
                for image_file in image_files:
                    try:
                        image = Image.open(image_file).convert("RGB")  # ✅ RGB 변환 (안전성 확보)
                        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device, torch.float16)

                        # ✅ 모델에 입력하여 캡션 생성
                        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
                        caption = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                        # ✅ 캡션을 파일에 저장 (이미지 파일명 없이 내용만 기록)
                        f.write(f"{caption}")
                        f.flush()  # 즉시 디스크에 기록

                    except Exception as e:
                        print(f"⚠️ Error processing {image_file}: {e}")
                        continue

            print(f"✅ Captions saved in: {output_file}")