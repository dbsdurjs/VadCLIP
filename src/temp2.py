import os
import csv

def rewrite_csv(old_csv, new_csv):
    """
    old_csv 파일을 읽어서, 
    '/home/.../aligned_output_xd/서브폴더/파일.npy,label'
    형태를
    '/home/.../aligned_output_xd/파일.npy,label'
    로 바꾼 뒤, new_csv에 저장.
    """
    with open(old_csv, 'r', newline='', encoding='utf-8') as fin, \
         open(new_csv, 'w', newline='', encoding='utf-8') as fout:
        
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        
        for row in reader:
            # row는 [path, label] 구조라고 가정
            if len(row) < 2:
                continue
            
            old_path, label = row[0], row[1]

            # 1) 경로에서 파일명 추출
            filename = os.path.basename(old_path)  
            # 예: "Bad.Boys.1995__#00-26-51_00-27-53_label_B2-0-0.npy"

            # 2) 상위 디렉터리 추출: 
            #    old_path = ".../aligned_output_xd/폴더/파일.npy"
            #    os.path.dirname(old_path) = ".../aligned_output_xd/폴더"
            #    os.path.dirname(os.path.dirname(old_path)) = ".../aligned_output_xd"
            upper_dir = os.path.dirname(os.path.dirname(old_path))
            # 예: ".../aligned_output_xd"

            # 3) 새 경로 구성
            new_path = os.path.join(upper_dir, filename)
            # 예: ".../aligned_output_xd/Bad.Boys.1995__#00-26-51_00-27-53_label_B2-0-0.npy"

            # 4) CSV에 기록
            writer.writerow([new_path, label])

if __name__ == "__main__":
    first_csv_path = "./list/xd_CLIP_rgbtest.csv"
    second_csv_path = "./list/xd_CLIP_rgbtest_caption(clean).csv"
    s_csv_path = "./list/xd_CLIP_rgbtest_caption_1.csv"

    # rewrite_csv(second_csv_path, s_csv_path)
    main(first_csv_path, second_csv_path)
