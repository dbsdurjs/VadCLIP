import os
import csv

def make_path(original_csv_path, text_feature_dir, new_csv_path):
    new_rows = []
    with open(original_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # 원본 이미지 feature 파일 경로
            img_feature_path = row['path']
            # 파일 이름 추출 (예: Bad.Boys.1995__#00-39-10_00-39-42_label_B2-0-0__1.npy)
            file_name = os.path.basename(img_feature_path)
            file_name = file_name.rsplit('__', 1)[0] + '.npy'
            # 텍스트 feature 파일 경로 생성
            text_feature_path = os.path.join(text_feature_dir, file_name)
            
            # 파일 이름이 같은 텍스트 feature 파일이 존재하는 경우에만 추가
            if os.path.exists(text_feature_path):
                new_rows.append({'path': text_feature_path, 'label': row['label']})
            else:
                print(f"파일 {text_feature_path}가 존재하지 않습니다.")

    # 새로운 CSV 파일로 저장
    with open(new_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['path', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"총 {len(new_rows)}개의 행이 {new_csv_path} 파일에 저장되었습니다.")

if __name__ == '__main__':

    # 원본 이미지 feature CSV 파일 경로
    original_csv_path = "./list/xd_CLIP_rgbtest.csv"

    # train 텍스트 feature가 저장된 디렉토리 경로
    text_feature_dir = "../VAD_dataset/XD-Violence/xdclip_caption_feature(clean)"

    # 저장할 새로운 CSV 파일 경로
    new_csv_path = "./list/xd_CLIP_rgbtest_caption(clean).csv"
    make_path(original_csv_path, text_feature_dir, new_csv_path)