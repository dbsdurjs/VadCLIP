# 필요한 코드
# 캡션을 txt파일로 저장할 때 json파일로 같이 저장 + 숫자 : 캡션 내용 형식으로 저장

import os
import json

# txt 파일들이 있는 디렉토리 경로
directory = '../VAD_dataset/XD-Violence/Extracted_Frames_captions'

# 디렉토리 내 파일들을 순회합니다.
for filename in os.listdir(directory):
    # 확장자가 .txt 인 파일만 처리합니다.
    if filename.endswith('.txt'):
        txt_path = os.path.join(directory, filename)
        data = {}
        
        # txt 파일을 한 줄씩 읽습니다.
        with open(txt_path, 'r', encoding='utf-8') as txt_file:
            for line in txt_file:
                line = line.strip()
                # 빈 줄은 건너뜁니다.
                if not line:
                    continue
                
                # 첫 번째 ':'를 기준으로 키와 캡션을 분리합니다.
                if ':' in line:
                    key_part, caption = line.split(":", 1)
                    key_part = key_part.strip()
                    caption = caption.strip()
                    
                    # "frame_" 이후의 부분만 추출합니다.
                    # 예: "v=Gm73TwtUyGY__#1_label_G-0-0_frame_00000.jpg" -> "00000.jpg"
                    if "frame_" in key_part:
                        extracted_key = key_part.split("frame_")[-1]
                        # 확장자 ".jpg" 제거
                        if extracted_key.endswith(".jpg"):
                            extracted_key = extracted_key[:-4]
                    else:
                        extracted_key = key_part
                    
                    data[extracted_key] = caption
        
        # txt 파일 이름을 json 파일 이름으로 변경 (.txt -> .json)
        json_filename = filename.replace('.txt', '.json')
        json_path = os.path.join(directory, json_filename)
        
        # 구성한 데이터를 JSON 파일로 저장합니다.
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        
        print(f"{filename} 파일이 {json_filename} 파일로 변환되었습니다.")
