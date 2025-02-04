import os, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BlipProcessor, BlipForConditionalGeneration

if __name__ == '__main__':
    folder = "./Abuse001_x264"
    image_files = []
    i = 0
    while True:
        image_path = os.path.join(folder, f"frame_{i:05d}.jpg")
        if os.path.exists(image_path):
            image_files.append(image_path)
            i += 1
        else:
            break

    if not image_files:
        print("이미지가 없습니다.")
        exit()

    processor = AutoProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model = AutoModelForImageTextToText.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 각 이미지마다 캡션을 생성
    # 결과 캡션을 저장할 파일을 미리 열어둡니다.
    output_file = "individual_captions.txt"
    image_files = image_files[295:]
    with open(output_file, "w", encoding="utf-8") as f:
    # 각 이미지마다 캡션 생성 및 저장
        for image_file in image_files:
            image = Image.open(image_file)
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # 생성된 캡션 출력
            print(f"{image_file}: {caption}")
            
            # 캡션을 파일에 저장 (각 캡션 생성 시마다 기록)
            f.write(f"{image_file}: {caption}\n")
            f.flush()  # 즉시 디스크에 기록되도록 flush 호출

    print(f"생성된 캡션이 '{output_file}' 파일에 저장되었습니다.")