import os
import csv

# 기존 함수 재사용
def count_videos_by_class(root_dir, class_map, exts=('.mp4', '.avi', '.mov', '.mkv')):
    """
    root_dir: 동영상들이 모여 있는 디렉터리
    class_map: {파일명에 들어있는 키워드: 라벨}
    exts: 카운트할 동영상 확장자 튜플
    """
    all_files = [f for f in os.listdir(root_dir)
                 if os.path.isfile(os.path.join(root_dir, f))
                 and f.lower().endswith(exts)]

    counts = {}
    for key, label in class_map.items():
        cnt = sum(1 for f in all_files if key.lower() in f.lower())
        counts[label] = cnt

    return counts

if __name__ == "__main__":
    root = "../VAD_dataset/XD-Violence/train_videos"  # 실제 경로로 바꿔주세요.

    # 새로운 데이터셋 클래스 매핑
    class_map2 = {
        'A':  'normal',
        'B1': 'fighting',
        'B2': 'shooting',
        'B4': 'riot',
        'B5': 'abuse',
        'B6': 'car accident',
        'G':  'explosion'
    }

    counts2 = count_videos_by_class(root, class_map2)

    # 1) 결과 출력
    print("두번째 데이터셋 — 클래스별 동영상 개수:")
    for label, cnt in counts2.items():
        print(f"  {label}: {cnt}개")

    # 2) CSV로 저장
    out_csv = "class_video_counts_dataset2.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as wf:
        writer = csv.writer(wf)
        writer.writerow(["class", "count"])
        for label, cnt in counts2.items():
            writer.writerow([label, cnt])
    print(f"\n[class_video_counts_dataset2.csv] 로 저장되었습니다.")
